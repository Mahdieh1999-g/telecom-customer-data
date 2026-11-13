"""
Model Serving Module
TODO: Implement the endpoints below
"""
import time
import joblib
import json
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


app = FastAPI(title="ML Model Serving API", version="1.0.0")

MODEL = None
METADATA = None


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    latency: float


@app.on_event("startup")
async def startup_event():
    """Load model on startup.
    TODO: Load model.joblib and metadata.json from models/ directory
    """
    global MODEL, METADATA
    model_path = Path("models/model.joblib")
    metadata_path = Path("models/metadata.json")

    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found: models/model.joblib"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Metadata file not found: models/metadata.json"
        )

    MODEL = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        METADATA = json.load(f)


@app.get("/health")
def health_check():
    """Health check endpoint.
    TODO: Return status and model_version
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model_version": METADATA.get("model_version", "unknown")
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """Make a prediction.
    TODO: Implement prediction logic
    - Use MODEL.predict() and MODEL.predict_proba()
    - Log the prediction using log_prediction()
    - Return PredictionResponse
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    start_time = time.time()

    try:
        features = np.array(request.features).reshape(1, -1)

        prediction = MODEL.predict(features)[0]

        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 0.0

        latency = time.time() - start_time

        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            latency=latency
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
