
import pandas as pd
import numpy as np

from api.main import create_features, SENSOR_COLUMNS


def make_sample_data():
    """Create 12 cycles of sample engine sensor data."""

    rows = []

    for cycle in range(1, 13):
        row = {
            "unit": 1,
            "cycle": cycle,
            "setting_1": 0.1,
            "setting_2": 0.2,
            "setting_3": 0.3,
        }

        for i in range(1, 22):
            row[f"sensor_{i}"] = float(cycle * i)

        rows.append(row)

    return pd.DataFrame(rows)


def test_feature_count():
    df = make_sample_data()
    result = create_features(df)

    model_features = [
        col for col in result.columns
        if col not in ["unit", "RUL"]
    ]

    assert len(model_features) == 130


def test_sensor_difference():
    df = make_sample_data()

    result = create_features(df)

    # Sensor 1 increases by 1 each cycle
    assert result["sensor_1_diff"].iloc[0] == 0
    assert result["sensor_1_diff"].iloc[1] == 1


def test_rolling_mean():
    df = make_sample_data()

    result = create_features(df)

    # First five sensor_1 values: 1, 2, 3, 4, 5
    expected_mean = np.mean([1, 2, 3, 4, 5])

    assert np.isclose(
        result["sensor_1_mean_5"].iloc[4],
        expected_mean
    )


def test_rolling_standard_deviation():
    df = make_sample_data()

    result = create_features(df)

    expected_std = np.std([1, 2, 3, 4, 5], ddof=1)

    assert np.isclose(
        result["sensor_1_std_5"].iloc[4],
        expected_std
    )


def test_no_missing_feature_values():
    df = make_sample_data()

    result = create_features(df)

    assert not result[SENSOR_COLUMNS].isnull().any().any()

    engineered_columns = [
        col for col in result.columns
        if "_diff" in col or "_mean_" in col or "_std_" in col
    ]

    assert not result[engineered_columns].isnull().any().any()


def test_feature_order_is_preserved():
    df = make_sample_data()

    result = create_features(df)

    assert result["cycle"].tolist() == list(range(1, 13))
