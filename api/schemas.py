from pydantic import BaseModel, Field
from typing import List


class EngineCycle(BaseModel):
    cycle: int = Field(gt=0)

    setting_1: float
    setting_2: float
    setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    engine_id: int = Field(gt=0)
    history: List[EngineCycle] = Field(min_length=1)


class PredictionResponse(BaseModel):
    engine_id: int
    last_observed_cycle: int
    predicted_rul: float
    risk_level: str
    anomaly_score: float
    anomaly_detected: bool
    maintenance_alert: str