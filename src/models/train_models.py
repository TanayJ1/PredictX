import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "RUL"
ID_COLUMN = "unit"


def load_features():

    train_path = DATA_DIR / "train_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            "Run python -m src.features.build_features first."
        )

    df = pd.read_csv(train_path)

    if "cycle_progress" in df.columns:
        raise ValueError(
            "Remove cycle_progress from feature engineering "
            "and regenerate the datasets."
        )

    if df.isnull().any().any():
        raise ValueError("Missing values found in training features.")

    if TARGET not in df.columns:
        raise ValueError("RUL target column is missing.")

    feature_columns = [
        col for col in df.columns
        if col not in [TARGET, ID_COLUMN]
    ]

    X = df[feature_columns].astype(np.float32)
    y = df[TARGET].astype(np.float32)
    groups = df[ID_COLUMN]

    return X, y, groups, feature_columns


def nasa_score(actual, predicted):

    errors = predicted - actual

    scores = np.where(
        errors < 0,
        np.exp(-errors / 13) - 1,
        np.exp(errors / 10) - 1
    )

    return float(np.sum(scores))


def evaluate_model(name, model, X_val, y_val):

    predictions = model.predict(X_val)

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    score = nasa_score(
        y_val.to_numpy(),
        predictions
    )

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE:       {mae:.3f}")
    print(f"RMSE:      {rmse:.3f}")
    print(f"NASA Score: {score:.3f}")

    return {
        "name": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "NASA_Score": score,
        "model": model
    }


def train():

    X, y, groups, feature_columns = load_features()

    # Split entire engines, not individual rows.
    engine_ids = groups.unique()

    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=0.20,
        random_state=42
    )

    train_mask = groups.isin(train_ids)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    # Evaluate only the latest observation of each validation engine.
# Validation: simulate engines observed before failure.

    val_mask = groups.isin(val_ids)

    val_df = pd.read_csv(
    DATA_DIR / "train_features.csv"
    )

    rng = np.random.default_rng(42)

    validation_rows = []

    for unit in val_ids:

        engine = val_df[val_df["unit"] == unit].copy()

        final_cycle = engine["cycle"].max()

    # Randomly select an observation point
    # between 50% and 90% of engine lifetime.
        fraction = rng.uniform(0.50, 0.90)

        cutoff = max(1, int(final_cycle * fraction))

        observed = engine[engine["cycle"] <= cutoff]

        latest = observed.iloc[-1].copy()

        actual_rul = final_cycle - latest["cycle"]

        latest["RUL"] = min(actual_rul, 125)

        validation_rows.append(latest)

    validation = pd.DataFrame(validation_rows)

    X_val = validation[feature_columns].astype(np.float32)
    y_val = validation["RUL"].astype(np.float32)

    print("Total engines:", len(engine_ids))
    print("Training engines:", len(train_ids))
    print("Validation engines:", len(val_ids))
    print("Training rows:", len(X_train))
    print("Validation observations:", len(X_val))
    print("Input features:", len(feature_columns))

    results = []

    # MODEL 1: Linear Regression
    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    print("\nTraining Linear Regression...")

    linear_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            "Linear Regression",
            linear_model,
            X_val,
            y_val
        )
    )

    # MODEL 2: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=5,
        max_samples=0.7,
        random_state=42,
        n_jobs=2
    )

    print("\nTraining Random Forest...")

    rf_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            "Random Forest",
            rf_model,
            X_val,
            y_val
        )
    )

    # MODEL 3: XGBoost
    xgb_configs = [
        {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05
        },
        {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.04
        }
    ]

    for index, config in enumerate(xgb_configs, start=1):

        xgb_model = XGBRegressor(
            **config,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=2,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            n_jobs=2
        )

        print(f"\nTraining XGBoost configuration {index}...")

        xgb_model.fit(X_train, y_train)

        results.append(
            evaluate_model(
                f"XGBoost Config {index}",
                xgb_model,
                X_val,
                y_val
            )
        )

    # Select the best model by validation MAE.
    best = min(results, key=lambda result: result["MAE"])

    print("\n" + "=" * 45)
    print("BEST MODEL:", best["name"])
    print("Validation MAE:", best["MAE"])
    print("=" * 45)

    # Save model and metadata.
    joblib.dump(
        best["model"],
        MODEL_DIR / "best_model.joblib"
    )

    with open(MODEL_DIR / "model_metadata.json", "w") as f:

        json.dump(
            {
                "model_name": best["name"],
                "features": feature_columns,
                "validation_MAE": best["MAE"],
                "validation_RMSE": best["RMSE"],
                "validation_NASA_Score": best["NASA_Score"]
            },
            f,
            indent=4
        )

    # Save model comparison.
    comparison = pd.DataFrame([
        {
            "Model": r["name"],
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "NASA_Score": r["NASA_Score"]
        }
        for r in results
    ])

    comparison = comparison.sort_values("MAE")

    comparison.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    # Feature importance for tree-based models.
    if hasattr(best["model"], "feature_importances_"):

        importance = pd.DataFrame({
            "feature": feature_columns,
            "importance": best["model"].feature_importances_
        })

        importance = importance.sort_values(
            "importance",
            ascending=False
        )

        importance.to_csv(
            MODEL_DIR / "feature_importance.csv",
            index=False
        )

        print("\nTOP 15 IMPORTANT FEATURES")
        print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    train()