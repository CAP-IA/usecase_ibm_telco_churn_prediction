from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from category_encoders import WOEEncoder
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
THRESHOLD = CONFIG["threshold"]

RAW_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

SERVICE_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",]

CHOICES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{name: ["No", "Yes", "No internet service"] for name in SERVICE_COLUMNS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",],
}


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Prepare raw customer fields and create features without using the target."""

    def fit (self, X, y=None):
        """Return self because this transformer learns no parameters."""
        return self
    
    def transform(self, X):
        """Validate columns, convert numeric fields, and create service features."""
        missing = sorted(set(RAW_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        data = X.loc[:, RAW_COLUMNS].copy()
        for name in ("tenure", "MonthlyCharges"):
            data[name] = pd.to_numeric(data[name], errors="raise")
        data["SeniorCitizen"] = pd.to_numeric(data["SeniorCitizen"], errors="raise").astype("Int64").astype("string")
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data["No_internet_service"] = data[SERVICE_COLUMNS].eq("No internet service").all(axis=1).astype(int)
        data["Internet_services_count"] = data[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
        data[SERVICE_COLUMNS] = data[SERVICE_COLUMNS].replace("No internet service", "No")
        return data


class ChurnModel(ClassifierMixin, BaseEstimator):
    """Fit WOE preprocessing and LightGBM as one model for raw customer data."""

    def __init__(self, model_params=None, threshold=THRESHOLD):
        self.model_params = model_params
        self.threshold = threshold

    def fit(self, X, y):
        """Fit all preprocessing steps and LightGBM using class-balanced weights."""
        fixed_params = {
            "boosting_type": "gbdt", "objective": "binary", "metric": "binary_logloss", "subsample_freq": 1, 
            "random_state": 42, "n_jobs": -1, "verbosity": -1, "deterministic": True, "force_col_wise": True,
        }
        params = {**fixed_params, **(self.model_params or {})}
        self.pipeline_ = Pipeline([
            ("features", FeatureBuilder()),
            ("woe", WOEEncoder(cols=CATEGORICAL_COLUMNS, handle_unknown="value", handle_missing="value")),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
            ("model", LGBMClassifier(**params)),
        ])
        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        if min(positives, negatives) == 0:
            raise ValueError("Both churn classes are required.")
        weights = np.where(np.asarray(y) == 1, negatives / positives, 1.0)
        self.pipeline_.fit(X, y, model__sample_weight=weights)
        self.classes_ = self.pipeline_.named_steps["model"].classes_
        return self

    def predict_proba(self, X):
        """Return the predicted probabilities for each class."""
        return self.pipeline_.predict_proba(X)

    def predict(self, X):
        """Predict churn (1) when its probability reaches the decision threshold."""
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def encoded_features(self, X):
        """Fitted WOE and imputation output for a SHAP explanation."""
        return self.pipeline_[:-1].transform(X)