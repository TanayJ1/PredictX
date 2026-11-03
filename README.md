# PredictX — Industrial Predictive Maintenance

PredictX is a machine learning-based predictive maintenance platform that estimates the Remaining Useful Life (RUL) of turbofan engines using sensor and operating-cycle data.

## Features

- Remaining Useful Life prediction using machine learning.
- Engine anomaly detection using Isolation Forest.
- SHAP-based model explainability.
- Feature engineering using sensor differences and rolling statistics.
- FastAPI prediction backend.
- Interactive Streamlit dashboard.
- Automated testing with pytest.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- FastAPI
- Streamlit
- Plotly
- Pytest

## Dataset

NASA C-MAPSS FD004 turbofan engine degradation dataset.

Dataset: https://zenodo.org/records/15346912

## Model Evaluation

Selected model: Linear Regression.

| Metric | Official Test Result |
|---|---:|
| MAE | 26.53 cycles |
| RMSE | 33.10 cycles |
| NASA Score | 10054.03 |

The model was selected using a separate validation dataset and evaluated on the official FD004 test set.

## Anomaly Detection

Isolation Forest identifies unusual engine operating patterns. Anomaly flags do not necessarily indicate an actual failure.

## Architecture

NASA C-MAPSS Dataset → Feature Engineering → ML Model → FastAPI → Streamlit Dashboard

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn api.main:app --reload
```

Start the dashboard in another terminal:

```bash
streamlit run dashboard/app.py
```

Run tests:

```bash
python -m pytest -v
```

## Limitations

- Predictions are estimates, not exact failure dates.
- Risk categories use heuristic RUL thresholds.
- Anomaly detection does not prove equipment failure.
- The project uses simulated NASA turbofan data, not live industrial telemetry.
