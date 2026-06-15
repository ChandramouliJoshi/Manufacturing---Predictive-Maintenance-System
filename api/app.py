from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.modeling import LEAKAGE_COLUMNS

MODEL_PATH = Path("models/production_failure_model.joblib")
TARGET_COLUMN = "machine_failure"

app = FastAPI(
    title="FactoryGuard AI Predictive Maintenance API",
    description="Predict failure probability for manufacturing equipment using the trained FactoryGuard AI model.",
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Engineered feature values for predictive maintenance.")


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    model_status: str


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run the training pipeline first.")
    return joblib.load(MODEL_PATH)


@app.get("/")
def read_root():
    return {
        "service": "FactoryGuard AI Predictive Maintenance API",
        "status": "ready",
        "model_path": str(MODEL_PATH),
        "usage": "POST /predict with a JSON payload containing engineered feature values.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = load_model()
    feature_data = request.features.copy()

    if TARGET_COLUMN in feature_data:
        feature_data.pop(TARGET_COLUMN)

    cleaned = {
        key: float(value)
        for key, value in feature_data.items()
        if key not in LEAKAGE_COLUMNS
    }

    if not cleaned:
        raise HTTPException(status_code=400, detail="No valid feature columns were provided.")

    try:
        X = pd.DataFrame([cleaned])
        probability = float(model.predict_proba(X)[:, 1][0])
        prediction = int(probability >= 0.5)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictionResponse(
        failure_probability=probability,
        prediction=prediction,
        model_status="success",
    )
