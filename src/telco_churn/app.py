from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import gradio as gr

from .predict import load_model, predict
from.gradio_ui import GRADIO_HEAD, demo, newspaper_theme


app = FastAPI(
    title = "Telco Customer Churn Prediction API",
    description = "ML API for predicting customer churn in telecom industry",
    version = "1.0.0"
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model unavailable") from exc
    return {"status": "ready"}


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: Literal["0", "1"]
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def get_prediction(data: CustomerData):
    return {"prediction": predict(data.model_dump())}


app = gr.mount_gradio_app(
    app,
    demo,
    path="/churn-predictor_demo_cap-ia",
    theme=newspaper_theme,
    head=GRADIO_HEAD,
)