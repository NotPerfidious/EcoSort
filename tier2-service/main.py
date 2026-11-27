import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from clip_arbiter import CLIPArbiter

CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
arbiter: CLIPArbiter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global arbiter
    arbiter = CLIPArbiter(CLIP_MODEL_ID)
    yield


app = FastAPI(
    title="EcoSort Tier-2 Cloud Arbiter (OpenAI CLIP ViT-B/32)", lifespan=lifespan
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "tier2-clip-vit-b32",
        "model_loaded": arbiter is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if arbiter is None:
        raise HTTPException(status_code=503, detail="CLIP arbiter not initialized")
    raw_bytes = await file.read()
    try:
        return arbiter.predict(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CLIP inference error: {exc}")
