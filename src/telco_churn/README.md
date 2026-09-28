# Application and training code

This folder contains the code that trains the churn model and serves predictions. The [root README](../../README.md) covers installation, the dataset and Docker, the [notebook README](../../notebooks/README.md) describes the earlier model exploration.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| `data.py` | Load the CSV, strip column names, drop customer IDs, convert `TotalCharges` to numeric and fill missing numeric values with zero, encode `Churn` and create a stratified holdout split |
| `preprocessing.py` | Define the raw column list, categorical columns and dropdown choices. `FeatureBuilder` prepares customer fields and makes two service features, `ChurnModel` owns the fitted WOE encoder, median imputer, LightGBM classifier, class weights and decision threshold |
| `train.py` | Read `configs/train.yaml`, run the Optuna objective with stratified folds, refit and evaluate the selected model, log to MLflow and assign the registered version the `candidate` alias |
| `predict.py` | Load and cache a fitted MLflow model, build one customer row, return a prediction and probability and optionally compute local SHAP factors |
| `visualizations.py` | Build the Plotly risk gauge and SHAP contribution chart displayed in Gradio |
| `gradio_ui.py` | Define the customer form, examples, callback and visual output, load the optional remote theme with a local fallback |
| `app.py` | Define FastAPI routes, validate `/predict` requests with Pydantic and mount the Gradio interface |

The only training command is `python -m telco_churn.train`. No separate orchestration script is needed for the current pipeline.

## Training: where fitting happens

Run the training module from the **repository root**, with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m telco_churn.train
```

The CLI also accepts `--data`, `--trials` and `--folds`. Its defaults and the holdout fraction, random seed, threshold and MLflow names come from [`configs/train.yaml`](../../configs/train.yaml). For example:

```bash
PYTHONPATH=src python -m telco_churn.train --trials 30 --folds 5
```

`data.py` prepares the raw dataset and splits off the test set **before any supervised WOE fitting**. Optuna proposes LightGBM parameters, `objective()` creates a fresh `ChurnModel` for each training fold. `ChurnModel.fit()` fits `FeatureBuilder`, `WOEEncoder`, `SimpleImputer` and `LGBMClassifier` inside one scikit-learn pipeline. It also computes sample weights from that fold's class counts. The fold's validation features pass through the fitted pipeline without fitting on validation labels. Optuna maximizes average validation **recall** at the model's decision threshold.

After tuning, `main()` fits a new `ChurnModel` on all training rows with the best parameters. It evaluates the untouched holdout once, logs test metrics and the fitted pipeline to MLflow, registers a model version and assigns `candidate`. This training command does **not** replace the already exported [`model/`](../../model/) folder used by the app.

### What the fitted model contains

`FeatureBuilder` selects the expected raw features in a fixed order, converts numeric fields, represents `SeniorCitizen` as the strings `"0"`/`"1"` and creates:

- `No_internet_service`: `1` when all six internet service fields say `No internet service`
- `Internet_services_count`: number of those service fields whose value is `Yes`

It then replaces `No internet service` with `No` in those fields. WOE learns categorical encodings from the training labels, the imputer learns training medians, LightGBM learns the classifier. All three learned steps live in the saved pipeline. `ChurnModel.predict_proba()` returns probabilities and `ChurnModel.predict()` applies the stored decision threshold (currently `0.35`) to the probability of churn. Changing a YAML setting later does not alter an already fitted model.

## Inference: API and interactive demo

Start the application from the repository root:

```bash
PYTHONPATH=src uvicorn telco_churn.app:app --reload
```

| Route | What it returns |
| --- | --- |
| `GET /` | `{"status": "ok"}` if the API is running. |
| `GET /health/ready` | `{"status": "ready"}` if the model can load, HTTP `503` otherwise. |
| `POST /predict` | A readable prediction label, for example `{"prediction": "Likely to churn"}`. |
| `GET /docs` | Interactive FastAPI request documentation. |
| `GET /churn-predictor_demo_cap-ia/` | The Gradio customer form. |

`POST /predict` requires all 19 customer features. In particular, `SeniorCitizen` must be the **string** `"0"` or `"1"`, the Gradio form shows **No/Yes** but sends those strings. `tenure` is an integer, charges are numbers. `customerID` and `Churn` do not belong in a request. Here is a complete request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "No",
    "PaymentMethod": "Electronic check",
    "tenure": 35,
    "MonthlyCharges": 49.2,
    "TotalCharges": 1701.65
  }'
```

FastAPI passes the validated fields to `predict()`, which calls `predict_details()`. `predict_details()` builds a row in `RAW_COLUMNS` order and asks the loaded, fitted model for its probability and threshold. No encoding is refitted on the new customer. The API intentionally returns only the label, Gradio calls `predict_details(..., explain=True)` to display the probability, a gauge and an explanation chart.

The gauge compares the predicted probability with the stored threshold. The SHAP chart shows up to eight of the strongest factors from `explain_customer()`. Its positive or negative values describe contributions to the LightGBM model score (log-odds), not percentage-point changes in churn probability. Gradio also provides two example profiles to try.

### Model location and refresh

`predict.load_model()` uses `mlflow.sklearn.load_model()` and caches the result once per process. By default it reads [`model/`](../../model/), which contains the exported **complete fitted pipeline**, including WOE and LightGBM. Set `TELCO_MODEL_URI` to a different local MLflow directory or to a Registry URI reachable by the app. `MLFLOW_TRACKING_URI` and `MLFLOW_REGISTRY_URI` control the MLflow connection if a remote registry is used.

Training assigns `candidate` to a registered version, but it does not copy that version into `model/`, an app using the default local directory will continue to serve the previously exported version. Export the chosen model, check the tests against it, rebuild the Docker image and restart the running application to serve an updated snapshot.

## Tests

From the repository root, run `PYTHONPATH=src python -m pytest -q`. `tests/test_data.py` checks data helpers, `tests/test_visualizations.py` checks figures, `tests/test_model_contract.py` checks the exported model contract and `tests/test_api.py` checks FastAPI routes. These tests do not rerun Optuna or require the source CSV.
