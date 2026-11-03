from fastapi.testclient import TestClient

from api.main import app
from src.data.load_data import load_data

client = TestClient(app)

# Load actual NASA test data
_, test, _ = load_data()

# Use the complete history of the first engine
engine_id = int(test["unit"].iloc[0])

engine_history = (
    test[test["unit"] == engine_id]
    .sort_values("cycle")
)

# Prepare request payload
history = engine_history.drop(columns=["unit"]).to_dict(
    orient="records"
)

payload = {
    "engine_id": engine_id,
    "history": history
}

# Test health endpoint
health = client.get("/health")
print("HEALTH CHECK:", health.status_code, health.json())

# Test prediction endpoint
response = client.post("/predict", json=payload)

print("\nPREDICTION STATUS:", response.status_code)
print("PREDICTION RESULT:")
print(response.json())

assert response.status_code == 200
assert "predicted_rul" in response.json()

print("\nAPI TEST PASSED!")