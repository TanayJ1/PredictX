from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

app = FastAPI(
    title="PredictX API",
    description="Industrial Predictive Maintenance and Remaining Useful Life Prediction",
    version="1.0.0"
)

# Load models and metadata once when the API starts
rul_model = joblib.load(MODEL_DIR / "best_model.joblib")
anomaly_model = joblib.load(MODEL_DIR / "anomaly_detector.joblib")

with open(MODEL_DIR / "model_metadata.json") as f:
    model_metadata = json.load(f)

with open(MODEL_DIR / "anomaly_metadata.json") as f:
    anomaly_metadata = json.load(f)

FEATURES = model_metadata["features"]
ANOMALY_FEATURES = anomaly_metadata["features"]

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the project's training-time feature engineering."""

    df = df.copy()
    df = df.sort_values("cycle").reset_index(drop=True)

    for sensor in SENSOR_COLUMNS:
        df[f"{sensor}_diff"] = df[sensor].diff().fillna(0)

    for window in [5, 10]:
        for sensor in SENSOR_COLUMNS:
            df[f"{sensor}_mean_{window}"] = (
                df[sensor]
                .rolling(window=window, min_periods=1)
                .mean()
            )

            df[f"{sensor}_std_{window}"] = (
                df[sensor]
                .rolling(window=window, min_periods=2)
                .std()
                .fillna(0)
            )

    return df


def get_risk_level(rul: float) -> str:
    if rul <= 20:
        return "CRITICAL"
    elif rul <= 50:
        return "HIGH"
    elif rul <= 100:
        return "MODERATE"
    return "LOW"


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rul_model": model_metadata["model_name"],
        "anomaly_detector": "Isolation Forest",
        "features": len(FEATURES)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:
        # Convert submitted history into a DataFrame
        history = pd.DataFrame([
            cycle.model_dump() for cycle in request.history
        ])

        # Validate cycle order and uniqueness
        if history["cycle"].duplicated().any():
            raise HTTPException(
                status_code=400,
                detail="Duplicate cycle numbers are not allowed."
            )

        history = history.sort_values("cycle").reset_index(drop=True)

        if len(history) > 1:
            differences = history["cycle"].diff().dropna()
            if (differences <= 0).any():
                raise HTTPException(
                    status_code=400,
                    detail="Cycle numbers must be strictly increasing."
                )

        # Generate the exact training-time features
        featured = create_features(history)

        # Use latest observation
        latest = featured.iloc[[-1]].copy()

        # Verify all expected features exist
        missing = [f for f in FEATURES if f not in latest.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        X_rul = latest[FEATURES].astype(float)
        X_anomaly = latest[ANOMALY_FEATURES].astype(float)

        # Predict remaining useful life
        predicted_rul = float(rul_model.predict(X_rul)[0])
        predicted_rul = float(np.clip(predicted_rul, 0, 125))

        # Detect unusual sensor behavior
        anomaly_score = float(
            anomaly_model.decision_function(X_anomaly)[0]
        )

        threshold = float(anomaly_metadata["threshold"])
        anomaly_detected = anomaly_score < threshold

        risk_level = get_risk_level(predicted_rul)

        if risk_level == "CRITICAL":
            maintenance_alert = "URGENT INSPECTION RECOMMENDED"
        elif risk_level == "HIGH":
            maintenance_alert = "SCHEDULE MAINTENANCE"
        elif risk_level == "MODERATE":
            maintenance_alert = "MONITOR ENGINE CONDITION"
        else:
            maintenance_alert = "NO IMMEDIATE ACTION"

        if anomaly_detected:
            maintenance_alert += " - ANOMALY DETECTED"

        return PredictionResponse(
            engine_id=request.engine_id,
            last_observed_cycle=int(latest["cycle"].iloc[0]),
            predicted_rul=round(predicted_rul, 2),
            risk_level=risk_level,
            anomaly_score=round(anomaly_score, 5),
            anomaly_detected=bool(anomaly_detected),
            maintenance_alert=maintenance_alert
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))