import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

TARGET = "RUL"
ID_COLUMN = "unit"


def train_anomaly_detector():

    train = pd.read_csv(
        DATA_DIR / "train_features.csv"
    )

    # Identify sensor and engineered features.
    features = [
        col for col in train.columns
        if col not in [TARGET, ID_COLUMN]
    ]

    # Use relatively healthy training observations
    # to establish normal operating behavior.
    healthy = train[train[TARGET] >= 100].copy()

    if len(healthy) < 256:
        raise ValueError(
            "Not enough healthy observations to train detector."
        )

    X_healthy = healthy[features].astype("float32")

    # Train on a representative subset to control memory.
    X_fit = X_healthy.sample(
        n=min(5000, len(X_healthy)),
        random_state=42
    )

    detector = IsolationForest(
        n_estimators=100,
        max_samples=min(256, len(X_fit)),
        contamination="auto",
        random_state=42,
        n_jobs=2
    )

    detector.fit(X_fit)

    # Lower decision_function values indicate more abnormal data.
    reference_scores = detector.decision_function(X_healthy)

    # Calibrate threshold using the lower 5th percentile.
    threshold = float(
        np.percentile(reference_scores, 5)
    )

    joblib.dump(
        detector,
        MODEL_DIR / "anomaly_detector.joblib"
    )

    metadata = {
        "features": features,
        "threshold": threshold,
        "reference_observations": len(X_healthy),
        "contamination": "auto",
        "calibration_percentile": 5
    }

    with open(
        MODEL_DIR / "anomaly_metadata.json",
        "w"
    ) as f:

        json.dump(metadata, f, indent=4)

    print("Anomaly detector trained successfully!")
    print("Healthy reference observations:", len(X_healthy))
    print("Anomaly threshold:", threshold)


if __name__ == "__main__":
    train_anomaly_detector()