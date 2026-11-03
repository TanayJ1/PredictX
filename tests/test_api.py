
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.data.load_data import load_data


client = TestClient(app)


@pytest.fixture(scope="module")
def engine_payload():
    _, test_data, _ = load_data()

    engine_id = int(test_data["unit"].iloc[0])

    engine = (
        test_data[test_data["unit"] == engine_id]
        .sort_values("cycle")
        .drop(columns=["unit"])
    )

    history = []

    for record in engine.to_dict(orient="records"):
        history.append({
            key: int(value) if key == "cycle" else float(value)
            for key, value in record.items()
        })

    return {
        "engine_id": engine_id,
        "history": history
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["features"] == 130


def test_prediction_endpoint(engine_payload):
    response = client.post("/predict", json=engine_payload)

    assert response.status_code == 200

    result = response.json()

    assert result["engine_id"] == engine_payload["engine_id"]
    assert 0 <= result["predicted_rul"] <= 125
    assert result["risk_level"] in [
        "CRITICAL", "HIGH", "MODERATE", "LOW"
    ]
    assert isinstance(result["anomaly_detected"], bool)
    assert isinstance(result["maintenance_alert"], str)


def test_prediction_rejects_duplicate_cycles(engine_payload):
    payload = {
        "engine_id": engine_payload["engine_id"],
        "history": [
            dict(record)
            for record in engine_payload["history"]
        ]
    }

    # Make the second cycle duplicate the first
    payload["history"][1]["cycle"] = (
        payload["history"][0]["cycle"]
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "Duplicate cycle" in response.json()["detail"]


def test_prediction_rejects_empty_history():
    payload = {
        "engine_id": 1,
        "history": []
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
