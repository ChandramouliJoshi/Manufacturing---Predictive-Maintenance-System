from __future__ import annotations

from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.feature_engineering import FeatureConfig, SENSOR_COLUMNS, create_time_series_features
from src.modeling import LEAKAGE_COLUMNS

MODEL_PATH = Path("models/production_failure_model.joblib")
TARGET_COLUMN = "machine_failure"

FEATURE_ALIASES = {
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

app = FastAPI(
    title="FactoryGuard AI Predictive Maintenance API",
    description="Predict failure probability for manufacturing equipment using the trained FactoryGuard AI model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    model_status: str


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run the training pipeline first.")
    return joblib.load(MODEL_PATH)


def normalize_payload(payload: dict[str, Any]) -> dict[str, float]:
    """Accept nested, flat, and common CSV-style sensor payloads."""
    raw_features = payload.get("features", payload)
    if not isinstance(raw_features, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object or contain a 'features' object.",
        )

    normalized: dict[str, float] = {}
    for key, value in raw_features.items():
        canonical_key = FEATURE_ALIASES.get(key, key)
        if canonical_key in SENSOR_COLUMNS:
            try:
                normalized[canonical_key] = float(value)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Sensor value '{canonical_key}' must be numeric.",
                )

    missing = [col for col in SENSOR_COLUMNS if col not in normalized]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing raw sensor values: "
                f"{', '.join(missing)}. Send either {{'features': ...}} or a flat JSON object."
            ),
        )

    return normalized


@app.get("/")
def read_root():
    return {
        "service": "FactoryGuard AI Predictive Maintenance API",
        "status": "ready",
        "model_path": str(MODEL_PATH),
        "usage": "POST /predict with a JSON payload containing raw sensor feature values.",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    try:
        model = load_model()
        try:
            payload = await request.json()
        except JSONDecodeError:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        raw_features = normalize_payload(payload)

        raw_df = pd.DataFrame([raw_features])
        config = FeatureConfig()
        engineered_df = create_time_series_features(raw_df, config)

        # Get expected columns from model
        expected_cols = list(model.named_steps['imputer'].feature_names_in_)

        # Reorder to match model's training schema
        X = engineered_df[expected_cols]

        # Fill any remaining NaNs
        X = X.ffill(axis=0).bfill(axis=0)

        probability = float(model.predict_proba(X)[:, 1][0])
        prediction = int(probability >= 0.5)

        return PredictionResponse(
            failure_probability=probability,
            prediction=prediction,
            model_status="success",
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing feature column: {str(e)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")
