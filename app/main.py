"""
app/main.py
FastAPI inference server for SLT model.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

import sys
sys.path.insert(0, '..')

from src.inference import SignLanguageTranslator
from src.utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Sign Language Translation API",
    description="Translate sign language video to text gloss",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model once
try:
    translator = SignLanguageTranslator(
        checkpoint_path="../models/best_model.pt",
        base_config_path="../configs/base_config.yaml",
        data_config_path="../configs/data_config.yaml",
    )
    MODEL_READY = True
    logger.info("[API] Model initialized successfully")
except Exception as e:
    MODEL_READY = False
    logger.error(f"[API] Failed to initialize model: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if MODEL_READY else "error",
        "model": "slt_hybrid_ssl",
        "version": "1.0",
        "gpu": "available" if torch.cuda.is_available() else "unavailable",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict sign language gloss from video.

    Args:
        file: MP4 video file

    Returns:
        JSON with gloss and confidence
    """
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Only video files supported")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"[API] Processing: {file.filename}")
        result = translator.predict_from_video(tmp_path)
        return JSONResponse(result)

    except Exception as e:
        logger.error(f"[API] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink()


@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Batch predict on multiple videos.

    Returns:
        List of predictions
    """
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = translator.predict_from_video(tmp_path)
            results.append({"file": file.filename, **result})
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
        finally:
            Path(tmp_path).unlink()

    return JSONResponse(results)


@app.get("/")
async def root():
    """API documentation."""
    return {
        "name": "Sign Language Translation API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /predict": "Predict gloss from single video",
            "POST /batch_predict": "Predict gloss from multiple videos",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
