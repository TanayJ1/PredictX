from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

# Load final model and metadata
model = joblib.load(MODEL_DIR / "best_model.joblib")

with open(MODEL_DIR / "model_metadata.json", "r") as f:
    metadata = json.load(f)

features = metadata["features"]

train_df = pd.read_csv(DATA_DIR / "train_features.csv")
test_df = pd.read_csv(DATA_DIR / "test_features.csv")

X_train = train_df[features].copy()
X_test = test_df[features].copy()

# Handle both a pipeline and a standalone estimator
if hasattr(model, "named_steps"):
    estimator = model.steps[-1][1]
    scaler = model[:-1]

    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
else:
    estimator = model
    X_train_transformed = X_train
    X_test_transformed = X_test

# Explain the final model
if hasattr(estimator, "coef_"):
    # Linear model: SHAP explanations on the transformed features
    background = X_train_transformed
    if len(background) > 500:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(background), 500, replace=False)
        background = background[indices]

    explainer = shap.LinearExplainer(estimator, background)

elif hasattr(estimator, "feature_importances_"):
    # Tree-based model fallback
    explainer = shap.TreeExplainer(estimator)

else:
    raise ValueError("Unsupported model type for this explanation script.")

# Explain a sample of official test engines
sample_size = min(100, len(X_test_transformed))
X_sample = X_test_transformed[:sample_size]

shap_values = explainer(X_sample)

# Summary plot
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_sample,
    feature_names=features,
    show=False
)
plt.tight_layout()
plt.savefig(MODEL_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

# Overall feature impact
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_sample,
    feature_names=features,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(MODEL_DIR / "shap_importance.png", dpi=200, bbox_inches="tight")
plt.close()

# Explain one individual engine
plt.figure()
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.tight_layout()
plt.savefig(MODEL_DIR / "shap_individual.png", dpi=200, bbox_inches="tight")
plt.close()

# Save linear model coefficients when available
if hasattr(estimator, "coef_"):
    coefficients = pd.DataFrame({
        "feature": features,
        "coefficient": np.asarray(estimator.coef_).ravel()
    })
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values(
        "absolute_coefficient", ascending=False
    )
    coefficients.to_csv(
        MODEL_DIR / "linear_coefficients.csv",
        index=False
    )

print("SHAP explainability completed.")
print("Saved:")
print("- models/shap_summary.png")
print("- models/shap_importance.png")
print("- models/shap_individual.png")

if hasattr(estimator, "coef_"):
    print("- models/linear_coefficients.csv")