
<h1 align="center">Telco Customer Churn Predictor</h1>

<p align="center">
  An end-to-end MLOps application for predicting customer churn
  with explainable machine learning.
</p>

<p align="center">
  <a href="https://usecase-telco-churn-prediction-latest.onrender.com/churn-predictor_demo_cap-ia/">
    <img src="https://img.shields.io/badge/🚀_Launch_the_App-0078D4?style=for-the-badge"
         alt="Launch the App" height="32">
  </a>
</p>

<p align="center">
  <sub>Click above to access the interactive application</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Gradio-F97316?style=flat-square&logo=gradio&logoColor=white" alt="Gradio">
  <img src="https://img.shields.io/badge/LightGBM-167D5A?style=flat-square" alt="LightGBM">
  <img src="https://img.shields.io/badge/Optuna-1C6DBA?style=flat-square&logo=optuna&logoColor=white" alt="Optuna">
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/MLOps-334155?style=flat-square" alt="MLOps">
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" alt="GitHub Actions">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/SHAP-7B61FF?style=flat-square" alt="SHAP">
</p>

<hr>

An end-to-end MLOps project that predicts customer churn from telecommunications account and service data. The training pipeline uses Weight of Evidence (WOE) encoding for categorical features, optimizes LightGBM hyperparameters with Optuna, and uses MLflow to track training runs and register the model. For predictions, the application loads the fitted preprocessing and model to estimate a customer’s probability of churn. FastAPI provides the prediction API and hosts an interactive Gradio demo.

GitHub Actions runs the tests, builds a Docker image from the project’s Dockerfile and pushes it to Docker Hub. Render deploys that image to serve the application. Click the linked title at the top of this README to open the live demo.

This project is an educational demonstration based on the Telco Customer Churn dataset. Its predictions are not intended to guide decisions about individual customers.


## What is in the repository?

| Path | Purpose |
| --- | --- |
| [`notebooks/`](notebooks/README.md) | Exploratory analysis, model comparison and the reasons for choosing LightGBM and the decision threshold |
| [`src/telco_churn/`](src/telco_churn/README.md) | Reproducible training pipeline, prediction functions, FastAPI endpoints, Gradio interface and charts |
| [`configs/train.yaml`](configs/train.yaml) | Local data path, training settings, decision threshold and MLflow names |
| `model/` | Exported, fitted MLflow model served by the application and included in the Docker image |
| `tests/` | Unit tests and checks of the API and the bundled model |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Run tests on pushes to `main`, build and push the Docker image only when tests pass |
| [`dockerfile`](dockerfile) | Build the FastAPI/Gradio image with the exported model |

The raw CSV, MLflow tracking database and training runs are local files excluded from Git. The exported serving model in `model/` is included in Git.

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data). Download the CSV and save it as:

```text
data/external/Telco-Customer-Churn.csv
```

The CSV includes the `Churn` target (`Yes` or `No`), a customer identifier, account information, subscribed services and charges. Training removes the identifier and maps the target to `1` (churn) or `0` (no churn). The app accepts the **19 customer feature fields**, not the identifier or target. For exact API fields, see the [application README](src/telco_churn/README.md).

You need the CSV to retrain or run the entire notebook. You do **not** need it to serve the model already present in `model/`.

## Run the application locally

From the repository root, use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src uvicorn telco_churn.app:app --reload
```

Open:

- API documentation: <http://127.0.0.1:8000/docs>
- Interactive demo: <http://127.0.0.1:8000/churn-predictor_demo_cap-ia/>
- Model readiness: <http://127.0.0.1:8000/health/ready>

The app loads `model/` by default. Set `TELCO_MODEL_URI` before starting it to load a different local MLflow model directory or a reachable Registry URI. The model is cached after its first load, so restart the app after changing the artifact or its URI. See the [application README](src/telco_churn/README.md) for a sample `/predict` request.

## Train a new model

Put the CSV at the path above, then run from the repository root:

```bash
PYTHONPATH=src python -m telco_churn.train
```

You can override the CSV path, number of Optuna trials and cross-validation folds:

```bash
PYTHONPATH=src python -m telco_churn.train --data data/external/Telco-Customer-Churn.csv --trials 30 --folds 5
```

These defaults come from [`configs/train.yaml`](configs/train.yaml):

| Setting | Default | Used for |
| --- | --- | --- |
| `data_path` | `data/external/Telco-Customer-Churn.csv` | Input CSV when `--data` is omitted |
| `test_size` | `0.2` | Fraction reserved for the final test set |
| `random_state` | `42` | Train/test split, cross-validation shuffling and Optuna sampler |
| `threshold` | `0.35` | Churn decision rule stored in the fitted `ChurnModel` |
| `trials`, `folds` | `30`, `5` | Default Optuna trials and stratified folds |
| `experiment_name`, `model_name` | `telco_churn_lightgbm`, `telco_churn_lightgbm_woe` | MLflow experiment and registered model |

Training follows this order:

1. Load and clean the CSV, separate `Churn`, then make a stratified train/test split.
2. For each Optuna trial, evaluate its LightGBM parameters with stratified cross-validation **on the training set only**. Each fold fits a new feature builder, WOE encoder, imputer and LightGBM on that fold's training rows. The validation rows are transformed with that fitted encoder and used to score recall at the configured threshold.
3. Select the parameters with the highest mean cross-validation recall. Fit a fresh complete pipeline on **all training rows**.
4. Evaluate it once on the untouched test set: recall, precision, F1, ROC AUC and PR AUC. Log parameters, metrics and the fitted model to MLflow, register a new model version and set its `candidate` alias.

WOE uses the target, so fitting it before a CV split would leak validation labels into training. The final WOE encoder and classifier are fitted after tuning, using the training set only. The test set does not fit preprocessing or guide the Optuna search.

### MLflow and the model used by the app

Unless you set `MLFLOW_TRACKING_URI`, training uses a local `mlflow.db` in the repository root. `MLFLOW_REGISTRY_URI` can override the registry location, otherwise it uses the same URI. To inspect local experiments in a browser, start a server **from the repository root** using the same database:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Training registers a version under `telco_churn_lightgbm_woe` and moves its `candidate` alias. **The app and Docker image do not switch to that version automatically:** by default they use the exported snapshot in `model/`. To deploy another version, export the selected MLflow model into `model/`, rerun the tests and rebuild the image. Alternatively, point `TELCO_MODEL_URI` to a Registry URI accessible from the running app. `model/registered_model_meta` records the registry version associated with the current export.

## Tests and Docker

Run the tests from the repository root:

```bash
PYTHONPATH=src GRADIO_ANALYTICS_ENABLED=False python -m pytest -q
```

Tests cover data preparation, visualization functions, the FastAPI endpoints and compatibility of the exported `model/` with the app's input and feature order. They load the existing exported model, running tests does not train a new one or require the training CSV.

To build and run locally:

```bash
docker build -f dockerfile -t telco-churn:local .
docker run --rm -p 8000:8000 telco-churn:local
```

The Docker image includes `model/` and serves the API and Gradio interface on port `8000`. On a push to `main`, GitHub Actions runs the tests first, if they pass, it builds and pushes `cap-ia/usecases_capia:latest` to Docker Hub. The workflow requires GitHub secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`. It does not retrain the model or redeploy a running service.

## Further reading

- [How training, inference, the API and charts work](src/telco_churn/README.md)
- [How to explore the notebook and its original experiments](notebooks/README.md)
- [MIT licence](LICENSE)

Copyright © 2026 [CAP IA](https://www.u-bordeaux.fr/universite/notre-strategie/nos-leviers/cma-competences-et-metiers-davenir/cap-ia)
