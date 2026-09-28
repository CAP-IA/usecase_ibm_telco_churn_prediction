from pathlib import Path

import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd

from telco_churn.preprocessing import THRESHOLD


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_model_accepts_the_api_input(sample_customer):
    raw = pd.DataFrame([sample_customer])
    model_path = str(ROOT / "model")

    pyfunc = mlflow.pyfunc.load_model(model_path)
    assert int(pyfunc.predict(raw)[0]) in (0, 1)

    model = mlflow.sklearn.load_model(model_path)
    assert "SeniorCitizen" in model.pipeline_.named_steps["woe"].cols
    assert model.threshold == THRESHOLD

    encoded = model.encoded_features(raw)
    assert list(encoded.columns) == list(model.pipeline_.named_steps["model"].feature_name_)