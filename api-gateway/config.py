import os
import threading
from typing import Any, Dict

from pydantic import BaseModel, Field


class RuntimeConfigUpdate(BaseModel):
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    sla_timeout_ms: int | None = Field(default=None, ge=50, le=30000)
    shadow_enabled: bool | None = None
    active_learning_enabled: bool | None = None
    safe_default_category: str | None = Field(
        default=None, pattern="^(Recycle|Compost|Landfill)$"
    )
    circuit_breaker_failure_threshold: int | None = Field(default=None, ge=1, le=20)
    circuit_breaker_recovery_sec: float | None = Field(default=None, ge=1.0, le=300.0)


class RuntimeConfigStore:
    """Thread-safe in-memory runtime configuration supporting zero-downtime hot reload."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")),
            "sla_timeout_ms": int(os.getenv("SLA_TIMEOUT_MS", "1500")),
            "shadow_enabled": os.getenv("SHADOW_ENABLED", "true").lower() == "true",
            "active_learning_enabled": os.getenv(
                "ACTIVE_LEARNING_ENABLED", "true"
            ).lower()
            == "true",
            "safe_default_category": os.getenv("SAFE_DEFAULT_CATEGORY", "Landfill"),
            "circuit_breaker_failure_threshold": int(
                os.getenv("CB_FAILURE_THRESHOLD", "3")
            ),
            "circuit_breaker_recovery_sec": float(os.getenv("CB_RECOVERY_SEC", "30.0")),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def update(self, patch: RuntimeConfigUpdate) -> Dict[str, Any]:
        updates = patch.model_dump(exclude_none=True)
        with self._lock:
            self._state.update(updates)
            return dict(self._state)
