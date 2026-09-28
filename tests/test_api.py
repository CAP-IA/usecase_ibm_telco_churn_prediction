from pathlib import Path
from unittest.mock import patch

import gradio as gr
import pandas as pd
import pytest
from fastapi.testclient import TestClient

with patch.object(gr.Theme, "from_hub", return_value=gr.themes.Soft()):
    from telco_churn import app as app_module

from telco_churn.predict import load_model


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app_module.app)


@pytest.fixture(autouse=True)
def use_bundled_model(monkeypatch):
    monkeypatch.setenv("TELCO_MODEL_URI", str(ROOT / "model"))
    load_model.cache_clear()
    yield
    load_model.cache_clear()


def test_prediction_matches_bundled_model(sample_customer):
    response = client.post("/predict", json=sample_customer)
    model = load_model()
    probability = float(model.predict_proba(pd.DataFrame([sample_customer]))[0, 1])
    expected = "Likely to churn" if probability >= model.threshold else "Not likely to churn"

    assert response.status_code == 200
    assert response.json() == {"prediction": expected}


def test_invalid_senior_citizen_returns_422(sample_customer):
    response = client.post("/predict", json={**sample_customer, "SeniorCitizen": "maybe"})
    assert response.status_code == 422


def test_missing_fields_return_422():
    assert client.post("/predict", json={}).status_code == 422


def test_readiness_and_gradio_page():
    assert client.get("/health/ready").status_code == 200
    assert client.get("/churn-predictor_demo_cap-ia/").status_code == 200


def test_unavailable_model_returns_503(monkeypatch):
    def unavailable():
        raise FileNotFoundError("missing model")

    monkeypatch.setattr(app_module, "load_model", unavailable)
    assert client.get("/health/ready").status_code == 503 


def test_prediction_failure_does_not_return_success(monkeypatch, sample_customer):
    def unavailable(_values):
        raise RuntimeError("prediction unavailable")

    monkeypatch.setattr(app_module, "predict", unavailable)
    error_client = TestClient(app_module.app, raise_server_exceptions=False)
    assert error_client.post("/predict", json=sample_customer).status_code == 500