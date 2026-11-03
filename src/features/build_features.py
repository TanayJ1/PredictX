import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import load_data

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def add_rul_labels(df, max_rul=125):
    """
    Calculate Remaining Useful Life for every training cycle.
    RUL = Final engine cycle - Current engine cycle.
    """

    df = df.copy()

    max_cycles = df.groupby("unit")["cycle"].transform("max")

    df["RUL"] = (max_cycles - df["cycle"]).clip(upper=max_rul)

    return df


def create_features(df):
    df = df.copy()

    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    # 1. Sensor differences from previous cycle
    for sensor in SENSOR_COLUMNS:
        df[f"{sensor}_diff"] = (
            df.groupby("unit")[sensor].diff().fillna(0)
        )

    # 2. Rolling statistics
    for window in [5, 10]:

        for sensor in SENSOR_COLUMNS:

            grouped = df.groupby("unit")[sensor]

            df[f"{sensor}_mean_{window}"] = grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )

            df[f"{sensor}_std_{window}"] = grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=2
                ).std()
            ).fillna(0)

    # 3. Normalized engine age


    return df


def prepare_datasets():

    train, test, test_rul = load_data()

    # Add training labels
    train = add_rul_labels(train)

    # Engineer features independently for each dataset
    train_features = create_features(train)
    test_features = create_features(test)

    # Keep only the latest observation of each test engine
    test_features = (
        test_features
        .sort_values(["unit", "cycle"])
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )

    # Align actual test RUL labels with engine IDs
    test_units = sorted(test_features["unit"].unique())

    if len(test_units) != len(test_rul):
        raise ValueError(
            "Number of test engines does not match RUL labels."
        )

    test_features["RUL"] = test_rul["RUL"].to_numpy()

    # Save processed datasets
    train_features.to_csv(
        OUTPUT_DIR / "train_features.csv",
        index=False
    )

    test_features.to_csv(
        OUTPUT_DIR / "test_features.csv",
        index=False
    )

    print("Feature engineering completed!")
    print("Training features:", train_features.shape)
    print("Testing features:", test_features.shape)

    print("\nTraining RUL distribution:")
    print(train_features["RUL"].describe())

    print("\nTest RUL distribution:")
    print(test_features["RUL"].describe())


if __name__ == "__main__":
    prepare_datasets()