# Exploratory notebook

[`telco_churn_analysis_model_selection_optimization.ipynb`](telco_churn_analysis_model_selection_optimization.ipynb) records the analysis that led to the LightGBM pipeline in [`src/telco_churn/`](../src/telco_churn/README.md). This notebook is an exploratory record, which means the application does not execute it and the reproducible training entry point is `telco_churn.train`.

## What the notebook covers

1. Load the Telco Customer Churn CSV and explore the target, customer characteristics, services, tenure and billing. Plot customer churn patterns.
2. Clean columns, remove `customerID`, prepare numeric values and make a stratified training/test split.
3. Explore WOE encoding and multicollinearity (VIF). The encoder used for a held-out test set is fitted on training rows.
4. Compare **Random Forest, XGBoost, LightGBM, and CatBoost** with stratified cross-validation and decision thresholds `0.30`, `0.35` and `0.40`. The CV evaluation function refits WOE on each fold's training rows before transforming that fold's validation rows.
5. Use Optuna to tune CatBoost at threshold `0.40` and LightGBM at threshold `0.35` with recall as the optimization objective. Track experiments in MLflow.
6. Compare the recorded results, inspect LightGBM feature importance and select LightGBM with threshold `0.35` for the application.

The notebook's conclusion reports a LightGBM recall of `0.902` in cross-validation and `0.898` on its test set. These are **historical notebook results**, they are not a measurement of every later training run or proof of the version currently exported to `model/`. The application trains its own LightGBM pipeline using [`train.py`](../src/telco_churn/train.py).

The [`static/`](static/) directory contains two MLflow comparison images referenced by the notebook. Keep it beside the notebook so those images render.

## Run it locally

Download the [Telco Customer Churn CSV](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) to `data/external/Telco-Customer-Churn.csv` in the repository root. Use Python 3.12, install the root requirements and install the notebook-only tools it imports or needs to open the file:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In a separate terminal, start MLflow from the repository root if you want to run the notebook's experiment cells:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

The notebook hardcodes `http://127.0.0.1:5000` as its MLflow tracking URI, those cells need the server. The notebook reads the CSV using a relative `../data/external/...` path, so launch Jupyter with `notebooks/` as its working directory:

```bash
cd notebooks
jupyter notebook telco_churn_analysis_model_selection_optimization.ipynb
```

Running every cell trains several models and executes Optuna studies, so it can take substantially longer than opening the notebook to read its saved outputs. The `model/` snapshot served by the API is not replaced by running notebook cells. To train and register a new deployable pipeline, use the [training command in the root README](../README.md#train-a-new-model).
