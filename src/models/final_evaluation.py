import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "RUL"
ID = "unit"


def nasa_score(actual, predicted):

    errors = predicted - actual

    scores = np.where(
        errors < 0,
        np.exp(-errors / 13) - 1,
        np.exp(errors / 10) - 1
    )

    return float(scores.sum())


def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "NASA_Score": nasa_score(actual, predicted)
    }


def create_validation_samples(df, validation_ids):

    rng = np.random.default_rng(42)

    samples = []

    fractions = [0.50, 0.70, 0.85]

    for unit in validation_ids:

        engine = (
            df[df[ID] == unit]
            .sort_values("cycle")
        )

        final_cycle = engine["cycle"].max()

        for fraction in fractions:

            cutoff = max(
                1,
                int(final_cycle * fraction)
            )

            observed = engine[
                engine["cycle"] <= cutoff
            ]

            if observed.empty:
                continue

            latest = observed.iloc[-1].copy()

            actual_rul = final_cycle - latest["cycle"]

            latest[TARGET] = min(actual_rul, 125)

            samples.append(latest)

    return pd.DataFrame(samples)


def get_models():

    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            max_samples=0.7,
            random_state=42,
            n_jobs=2
        ),

        "XGBoost 1": XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=2,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=2
        ),

        "XGBoost 2": XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=2,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=2
        ),

        "XGBoost 3": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=5,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=2
        )
    }


def main():

    print("Loading datasets...")

    train = pd.read_csv(
        DATA_DIR / "train_features.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test_features.csv"
    )

    if "cycle_progress" in train.columns:
        raise ValueError("Remove cycle_progress before training.")

    features = [
        col for col in train.columns
        if col not in [ID, TARGET]
    ]

    if features != [
        col for col in test.columns
        if col not in [ID, TARGET]
    ]:
        raise ValueError("Train and test features do not match.")

    if train[features].isnull().any().any():
        raise ValueError("Missing values in training features.")

    if test[features].isnull().any().any():
        raise ValueError("Missing values in test features.")

    # ---------------------------------
    # 1. Engine-level validation split
    # ---------------------------------

    engine_ids = train[ID].unique()

    train_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=0.20,
        random_state=42
    )

    training = train[train[ID].isin(train_ids)]

    validation = create_validation_samples(
        train,
        validation_ids
    )

    X_train = training[features].astype("float32")
    y_train = training[TARGET].astype("float32")

    X_val = validation[features].astype("float32")
    y_val = validation[TARGET].astype("float32")

    print("\nVALIDATION INFORMATION")
    print("Training engines:", len(train_ids))
    print("Validation engines:", len(validation_ids))
    print("Validation samples:", len(validation))

    # ---------------------------------
    # 2. Train and compare models
    # ---------------------------------

    results = []
    trained_models = {}

    for name, model in get_models().items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = np.clip(
            model.predict(X_val),
            0,
            125
        )

        metrics = calculate_metrics(
            y_val.to_numpy(),
            predictions
        )

        print("MAE:", round(metrics["MAE"], 3))
        print("RMSE:", round(metrics["RMSE"], 3))
        print(
            "NASA Score:",
            round(metrics["NASA_Score"], 3)
        )

        results.append({
            "Model": name,
            **metrics
        })

        trained_models[name] = model

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "NASA_Score"
    )

    comparison.to_csv(
        MODEL_DIR / "final_model_comparison.csv",
        index=False
    )

    print("\nVALIDATION MODEL COMPARISON")
    print(comparison.to_string(index=False))

    # Select model based on NASA score.
    best_name = comparison.iloc[0]["Model"]

    print("\nSELECTED MODEL:", best_name)

    # ---------------------------------
    # 3. Retrain on all training engines
    # ---------------------------------

    print("\nRetraining selected model on all training data...")

    final_model = get_models()[best_name]

    X_full = train[features].astype("float32")
    y_full = train[TARGET].astype("float32")

    final_model.fit(X_full, y_full)

    # ---------------------------------
    # 4. Official FD004 test evaluation
    # ---------------------------------

    X_test = test[features].astype("float32")
    y_test = test[TARGET].astype("float32")

    test_predictions = np.clip(
        final_model.predict(X_test),
        0,
        125
    )

    test_metrics = calculate_metrics(
        y_test.to_numpy(),
        test_predictions
    )

    print("\nOFFICIAL FD004 TEST RESULTS")
    print("----------------------------")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ---------------------------------
    # 5. Save final production model
    # ---------------------------------

    joblib.dump(
        final_model,
        MODEL_DIR / "best_model.joblib"
    )

    metadata = {
        "model_name": best_name,
        "features": features,
        "selection_metric": "NASA_Score",
        "validation_metrics": (
            comparison.iloc[0][
                ["MAE", "RMSE", "NASA_Score"]
            ].to_dict()
        ),
        "official_test_metrics": test_metrics,
        "prediction_cap": 125
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w"
    ) as f:

        json.dump(metadata, f, indent=4)

    # ---------------------------------
    # 6. Save feature importance
    # ---------------------------------

    if hasattr(final_model, "feature_importances_"):

        importance = pd.DataFrame({
            "feature": features,
            "importance": final_model.feature_importances_
        })

        importance = importance.sort_values(
            "importance",
            ascending=False
        )

        importance.to_csv(
            MODEL_DIR / "feature_importance.csv",
            index=False
        )

        print("\nTOP 15 FINAL FEATURES")
        print(importance.head(15).to_string(index=False))

    # ---------------------------------
    # 7. Save official test predictions
    # ---------------------------------

    predictions_df = pd.DataFrame({
        "engine_id": test[ID],
        "actual_RUL": y_test,
        "predicted_RUL": np.round(test_predictions, 2),
        "absolute_error": np.abs(
            y_test.to_numpy() - test_predictions
        )
    })

    predictions_df.to_csv(
        MODEL_DIR / "official_test_predictions.csv",
        index=False
    )

    print("\nFinal model saved successfully!")
    print("Official test predictions saved!")


if __name__ == "__main__":
    main()