import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from inference import Tier1Engine

FP32_MODEL_PATH = os.getenv("FP32_MODEL_PATH", "/models/mobilenetv4_fp32.onnx")
INT8_MODEL_PATH = os.getenv("INT8_MODEL_PATH", "/models/mobilenetv4_int8.onnx")

engine: Tier1Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = Tier1Engine(FP32_MODEL_PATH, INT8_MODEL_PATH)
    yield


app = FastAPI(title="EcoSort Tier-1 Edge Service (ONNX Runtime)", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "tier1-mobilenetv4-onnx",
        "fp32_loaded": engine is not None and engine.fp32_sess is not None,
        "int8_loaded": engine is not None and engine.int8_sess is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(default=True),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Tier-1 model not initialized")
    raw_bytes = await file.read()
    try:
        return engine.predict(raw_bytes, explain=explain, shadow=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}")


@app.post("/predict/shadow")
async def predict_shadow(file: UploadFile = File(...)):
    if engine is None or engine.int8_sess is None:
        raise HTTPException(status_code=503, detail="INT8 shadow model not loaded")
    raw_bytes = await file.read()
    try:
        return engine.predict(raw_bytes, explain=False, shadow=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Shadow inference error: {exc}")


@app.post("/reload")
async def reload_models():
    if engine is None:
        raise HTTPException(status_code=503, detail="Tier-1 engine unavailable")
    try:
        return engine.reload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")
