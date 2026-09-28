import os
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

import sys
from . import preprocessing as preprocessing_module
#sys.modules.setdefault("preprocessing", preprocessing_module)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URI = str(ROOT / "model")


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the fitted churn pipeline from MLflow.

    Uses TELCO_MODEL_URI when it is set, otherwise, loads the model registered under the "candidate" alias.

    Returns:
        The fitted model, including preprocessing and LightGBM.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", tracking_uri))
    sys.modules.setdefault("preprocessing", preprocessing_module)
    return mlflow.sklearn.load_model(os.getenv("TELCO_MODEL_URI", DEFAULT_URI))


def predict(input_dict: dict) -> str:
    """Predict churn for one customer and return a readable label.

    Args:
        input_dict: Raw customer values keyed by feature name.

    Returns:
        Either "Likely to churn" or "Not likely to churn".
    """

    details = predict_details(input_dict)
    return details["prediction"]


def predict_details(values: dict, explain: bool = False) -> dict:
    """Predict churn probability and optionally explain one customer.

    Args:
        values: Raw customer values keyed by feature name.
        explain: Whether to calculate SHAP contributions.

    Returns:
        A dictionary containing the prediction label, churn probability, decision threshold, and binary churn decision. 
        When explain is True, it also contains "top_factors".
    """

    model = load_model()
    raw = pd.DataFrame([values], columns=preprocessing_module.RAW_COLUMNS)
    raw["SeniorCitizen"] = (pd.to_numeric(raw["SeniorCitizen"], errors="raise").astype("Int64").astype("string"))
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0).astype(float)
    probability = float(model.predict_proba(raw)[0, 1])
    is_churn = probability >= model.threshold
    prediction = "Likely to churn" if is_churn else "Not likely to churn"
    result = {
        "prediction": prediction,
        "churn_probability": probability,
        "threshold": float(model.threshold),
        "is_churn": is_churn,
    }

    if explain:
        result["top_factors"] = explain_customer(model, raw)

    return result


def explain_customer(model, raw: pd.DataFrame) -> list[dict]:
    """Find the ten features with the largest impact on one prediction.

    Args:
        model: Fitted churn pipeline containing WOE and LightGBM.
        raw: DataFrame containing one customer's raw feature values.

    Returns:
        Up to ten dictionaries with "feature", "value", and "effect".
        Effects are SHAP contributions to the model score in log-odds,
        not changes in probability.
    """

    import shap

    encoded = model.encoded_features(raw)
    values = shap.TreeExplainer(model.pipeline_.named_steps["model"]).shap_values(encoded)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != encoded.shape:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    display_values = model.pipeline_.named_steps["features"].transform(raw).iloc[0]
    factors = [
        {
            "feature": name,
            "value": str(display_values[name]),
            "effect": float(effect),
        }
        for name, effect in zip(encoded.columns, values[0], strict=True)
    ]
    
    return sorted(factors, key=lambda row: abs(row["effect"]), reverse=True)[:10]