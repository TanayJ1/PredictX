import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def get_risk_level(rul):

    if rul <= 20:
        return "CRITICAL"

    elif rul <= 50:
        return "HIGH"

    elif rul <= 100:
        return "MODERATE"

    return "LOW"


def generate_report():

    # Load trained models
    rul_model = joblib.load(
        MODEL_DIR / "best_model.joblib"
    )

    anomaly_model = joblib.load(
        MODEL_DIR / "anomaly_detector.joblib"
    )

    # Load model metadata
    with open(MODEL_DIR / "model_metadata.json") as f:
        rul_metadata = json.load(f)

    with open(MODEL_DIR / "anomaly_metadata.json") as f:
        anomaly_metadata = json.load(f)

    features = rul_metadata["features"]

    # Load final observation of each test engine
    test = pd.read_csv(
        DATA_DIR / "test_features.csv"
    )

    X = test[features].astype("float32")

    # Predict remaining useful life
    predictions = rul_model.predict(X)

    predictions = np.clip(predictions, 0, 125)

    # Detect abnormal sensor behavior
    anomaly_scores = anomaly_model.decision_function(X)

    threshold = anomaly_metadata["threshold"]

    anomaly_flags = anomaly_scores < threshold

    # Generate report
    report = pd.DataFrame({
        "engine_id": test["unit"].to_numpy(),
        "last_observed_cycle": test["cycle"].to_numpy(),
        "predicted_RUL": np.round(predictions, 2),
        "anomaly_score": np.round(anomaly_scores, 5),
        "anomaly_detected": anomaly_flags
    })

    report["risk_level"] = report["predicted_RUL"].apply(
        get_risk_level
    )

    report["maintenance_alert"] = np.where(
        report["anomaly_detected"],
        "INVESTIGATE SENSOR ANOMALY",
        "NO ANOMALY FLAG"
    )

    output_path = MODEL_DIR / "engine_health_report.csv"

    report.to_csv(
        output_path,
        index=False
    )

    print("\nENGINE HEALTH REPORT")
    print(report.head(15).to_string(index=False))

    print("\nRISK DISTRIBUTION")
    print(report["risk_level"].value_counts())

    print("\nANOMALY DISTRIBUTION")
    print(report["anomaly_detected"].value_counts())

    print("\nReport saved:", output_path)


if __name__ == "__main__":
    generate_report()