import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from circuit_breaker import CircuitBreaker, CircuitState
from config import RuntimeConfigStore, RuntimeConfigUpdate
from drift_tracker import DriftTracker, extract_image_stats
from metrics import (
    CIRCUIT_BREAKER_STATE,
    CONFIDENCE_THRESHOLD_GAUGE,
    FALLBACK_TOTAL,
    REQUEST_LATENCY_MS,
    REQUESTS_TOTAL,
    SHADOW_AGREEMENT_TOTAL,
    SHADOW_LATENCY_DELTA_MS,
    SHADOW_REQUESTS_TOTAL,
    TIER_LATENCY_MS,
)
from s3_ingestor import ActiveLearningIngestor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecosort.gateway")

TIER1_URL = os.getenv("TIER1_URL", "http://tier1-service:8001").rstrip("/")
TIER2_URL = os.getenv("TIER2_URL", "http://tier2-service:8002").rstrip("/")
REF_STATS_PATH = os.getenv("REFERENCE_STATS_PATH", "/models/reference_stats.json")
TELEMETRY_PATH = os.getenv("TELEMETRY_PATH", "/reports/inference_telemetry.jsonl")
STATIC_DIR = Path(__file__).parent / "static"

config_store = RuntimeConfigStore()
circuit_breaker = CircuitBreaker(
    failure_threshold=config_store.snapshot()["circuit_breaker_failure_threshold"],
    recovery_timeout_sec=config_store.snapshot()["circuit_breaker_recovery_sec"],
)
drift_tracker = DriftTracker(
    reference_stats_path=REF_STATS_PATH, telemetry_path=TELEMETRY_PATH
)
s3_ingestor = ActiveLearningIngestor()
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    CIRCUIT_BREAKER_STATE.set(int(circuit_breaker.state))
    CONFIDENCE_THRESHOLD_GAUGE.set(config_store.snapshot()["confidence_threshold"])
    yield
    await http_client.aclose()


app = FastAPI(title="EcoSort Adaptive Multi-Tier Inference Gateway", lifespan=lifespan)


async def _mirror_shadow_traffic(
    raw_bytes: bytes,
    filename: str,
    primary_category: str,
    primary_latency_ms: float,
) -> None:
    """Fire-and-forget async mirror to INT8 MobileNetV4 shadow endpoint."""
    if http_client is None:
        return
    try:
        resp = await http_client.post(
            f"{TIER1_URL}/predict/shadow",
            files={"file": (filename, raw_bytes, "image/jpeg")},
            timeout=httpx.Timeout(3.0),
        )
        resp.raise_for_status()
        data = resp.json()
        SHADOW_REQUESTS_TOTAL.labels(status="ok").inc()
        agreed = "true" if data.get("category") == primary_category else "false"
        SHADOW_AGREEMENT_TOTAL.labels(agreed=agreed).inc()
        delta = round(float(data.get("latency_ms", 0.0)) - primary_latency_ms, 2)
        SHADOW_LATENCY_DELTA_MS.set(delta)
    except Exception as exc:
        SHADOW_REQUESTS_TOTAL.labels(status="error").inc()
        logger.debug("[shadow] Mirror call failed: %s", exc)


@app.get("/")
async def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI static/index.html not found")
    return FileResponse(index_file)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "api-gateway",
        "circuit_breaker_state": circuit_breaker.state.name,
        "s3_active_learning_configured": s3_ingestor.configured,
        "config": config_store.snapshot(),
    }


@app.get("/metrics")
async def metrics():
    CIRCUIT_BREAKER_STATE.set(int(circuit_breaker.state))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/config")
async def get_config():
    return config_store.snapshot()


@app.put("/admin/config")
async def update_config(patch: RuntimeConfigUpdate):
    updated = config_store.update(patch)
    circuit_breaker.reconfigure(
        failure_threshold=updated["circuit_breaker_failure_threshold"],
        recovery_timeout_sec=updated["circuit_breaker_recovery_sec"],
    )
    CONFIDENCE_THRESHOLD_GAUGE.set(updated["confidence_threshold"])
    return {"status": "updated", "config": updated}


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    explain: bool = Query(default=True),
) -> Dict[str, Any]:
    if http_client is None:
        raise HTTPException(status_code=503, detail="Gateway HTTP client not ready")

    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    raw_bytes = await file.read()
    filename = file.filename or f"{request_id}.jpg"
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")

    try:
        brightness, rgb_mean = extract_image_stats(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    cfg = config_store.snapshot()
    tau = float(cfg["confidence_threshold"])
    sla_sec = float(cfg["sla_timeout_ms"]) / 1000.0

    # 1. Tier-1 Fast-Path (Fine-tuned MobileNetV4 ONNX + Spatial CAM)
    try:
        t1_resp = await http_client.post(
            f"{TIER1_URL}/predict",
            params={"explain": str(explain).lower()},
            files={"file": (filename, raw_bytes, file.content_type or "image/jpeg")},
            timeout=httpx.Timeout(3.0),
        )
        t1_resp.raise_for_status()
        t1_data = t1_resp.json()
    except Exception as exc:
        REQUESTS_TOTAL.labels(category="error", tier_used="tier1", status="error").inc()
        raise HTTPException(status_code=502, detail=f"Tier-1 inference failed: {exc}")

    t1_latency = float(t1_data["latency_ms"])
    t1_conf = float(t1_data["confidence"])
    TIER_LATENCY_MS.labels(tier="tier1").observe(t1_latency)

    # Default response fields from Tier-1
    final_category = t1_data["category"]
    final_confidence = t1_conf
    final_probs = t1_data["probabilities"]
    tier_used = "tier1-mobilenetv4-onnx"
    escalated = False
    fallback_triggered = False
    t2_latency: Optional[float] = None
    active_learning_queued = False

    # 2. Confidence-Based Routing -> Escalate to Tier-2 CLIP if conf < tau
    if t1_conf < tau:
        escalated = True

        # Active Learning Flywheel: queue low-confidence frame to S3 asynchronously
        if cfg["active_learning_enabled"]:
            active_learning_queued = True
            background_tasks.add_task(
                s3_ingestor.ingest,
                request_id,
                raw_bytes,
                {
                    "request_id": request_id,
                    "filename": filename,
                    "tier1_category": t1_data["category"],
                    "tier1_confidence": t1_conf,
                    "threshold_tau": tau,
                    "brightness": brightness,
                },
            )

        # SLA Circuit Breaker around Tier-2 CLIP Arbiter
        if not circuit_breaker.allow_request():
            fallback_triggered = True
            final_category = cfg["safe_default_category"]
            final_confidence = t1_conf
            tier_used = "sla-fallback-safe-default"
            FALLBACK_TOTAL.labels(reason="circuit_open").inc()
        else:
            try:
                t2_resp = await http_client.post(
                    f"{TIER2_URL}/predict",
                    files={
                        "file": (filename, raw_bytes, file.content_type or "image/jpeg")
                    },
                    timeout=httpx.Timeout(sla_sec),
                )
                t2_resp.raise_for_status()
                t2_data = t2_resp.json()
                circuit_breaker.record_success()

                final_category = t2_data["category"]
                final_confidence = float(t2_data["confidence"])
                final_probs = t2_data["probabilities"]
                t2_latency = float(t2_data["latency_ms"])
                tier_used = "tier2-clip-vit-b32"
                TIER_LATENCY_MS.labels(tier="tier2").observe(t2_latency)
            except (httpx.TimeoutException, httpx.HTTPError, Exception) as exc:
                circuit_breaker.record_failure()
                fallback_triggered = True
                final_category = cfg["safe_default_category"]
                final_confidence = t1_conf
                tier_used = "sla-fallback-safe-default"
                reason = (
                    "sla_timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "tier2_error"
                )
                FALLBACK_TOTAL.labels(reason=reason).inc()
                logger.warning("[gateway] Tier-2 fallback (%s): %s", reason, exc)

    # 3. Asynchronous Shadow Traffic Mirroring (INT8 MobileNetV4)
    shadow_mirrored = bool(cfg["shadow_enabled"])
    if shadow_mirrored:
        background_tasks.add_task(
            _mirror_shadow_traffic, raw_bytes, filename, t1_data["category"], t1_latency
        )

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    cb_state = circuit_breaker.state
    CIRCUIT_BREAKER_STATE.set(int(cb_state))
    REQUEST_LATENCY_MS.observe(total_latency_ms)
    REQUESTS_TOTAL.labels(
        category=final_category, tier_used=tier_used, status="ok"
    ).inc()

    drift_snapshot = drift_tracker.record(
        request_id=request_id,
        brightness=brightness,
        rgb_mean=rgb_mean,
        tier1_confidence=t1_conf,
        final_category=final_category,
        tier_used=tier_used,
        escalated=escalated,
        fallback_triggered=fallback_triggered,
        total_latency_ms=total_latency_ms,
    )

    return {
        "request_id": request_id,
        "category": final_category,
        "confidence": round(final_confidence, 4),
        "probabilities": final_probs,
        "tier_used": tier_used,
        "escalated": escalated,
        "fallback_triggered": fallback_triggered,
        "circuit_breaker_state": cb_state.name,
        "confidence_threshold": tau,
        "latency_ms": {
            "total": total_latency_ms,
            "tier1": t1_latency,
            "tier2": t2_latency,
        },
        "active_learning_queued": active_learning_queued,
        "shadow_mirrored": shadow_mirrored,
        "drift_snapshot": drift_snapshot,
        "explanation": t1_data.get("explanation"),
    }
