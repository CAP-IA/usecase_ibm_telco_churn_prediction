import argparse
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data import load_raw_data, preprocess_data, extract_target, make_holdout_split
from .preprocessing import ChurnModel, RAW_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
DATA_PATH = ROOT / CONFIG["data_path"]
MODEL_NAME = CONFIG["model_name"]
EXPERIMENT_NAME = CONFIG["experiment_name"]
RANDOM_STATE = CONFIG["random_state"]
THRESHOLD = CONFIG["threshold"]

def objective(trial, X_train, y_train, folds):
    """"Evaluate one set of LightGBM parameters using stratified cross-validation.

    A new pipeline, including its WOE encoder, is fitted on the training portion of each fold. The validation portion is used only for scoring.

    Args:
        trial: Optuna trial used to suggest LightGBM hyperparameters.
        X_train: Customer features from the training set.
        y_train: Binary churn labels corresponding to X_train.
        folds: Number of cross-validation folds.

    Returns:
        Mean validation recall across the folds.
    """

    depth = trial.suggest_int("max_depth", 3, 8)
    params = {
        "max_depth": depth,
        "num_leaves": trial.suggest_int("num_leaves", 7, min(127, 2**depth - 1)),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }
    
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    recalls = []
    for train_indices, val_indices in cv.split(X_train, y_train):
        model = ChurnModel(model_params=params)
        model.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
        prediction = model.predict(X_train.iloc[val_indices])
        recalls.append(recall_score(y_train.iloc[val_indices], prediction))

    return float(np.mean(recalls))


def main():
    """Train, evaluate, and register the churn prediction pipeline.

    Reads the command-line options --data, --trials, and --folds. Splits off a test set, tunes hyperparameters on the training set, fits the 
    final pipeline on that training set, then evaluates it on the test set. Logs the results to MLflow and assigns the registered model version 
    the "candidate" alias.
    """

    parser = argparse.ArgumentParser(description="Train and register the LightGBM churn pipeline.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--trials", type=int, default=CONFIG["trials"])
    parser.add_argument("--folds", type=int, default=CONFIG["folds"])
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 2:
        parser.error("Use at least one Optuna trial and two CV folds.")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", tracking_uri))
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_raw_data(args.data)
    data = preprocess_data(data)
    X, y = extract_target(data)
    X_train, X_test, y_train, y_test = make_holdout_split(X, y, CONFIG["test_size"], RANDOM_STATE)

    with mlflow.start_run(run_name="lightgbm_woe_optuna") as run:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(lambda trial: objective(trial, X_train, y_train, args.folds), n_trials=args.trials)

        model = ChurnModel(model_params=study.best_params).fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)
        metrics = {
            "test_recall": recall_score(y_test, predictions),
            "test_precision": precision_score(y_test, predictions, zero_division=0),
            "test_f1": f1_score(y_test, predictions),
            "test_roc_auc": roc_auc_score(y_test, probabilities),
            "test_pr_auc": average_precision_score(y_test, probabilities),
            "cv_recall": study.best_value,
        }

        mlflow.log_params({"encoding": "WOE", "threshold": THRESHOLD, "folds": args.folds, "trials": args.trials, **study.best_params})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

        example = X_train.loc[:, RAW_COLUMNS].head(5).copy()
        example["SeniorCitizen"] = example["SeniorCitizen"].astype("string")
        example["TotalCharges"] = pd.to_numeric(example["TotalCharges"], errors="coerce")
        logged = mlflow.sklearn.log_model(
            sk_model=model,
            name="churn_pipeline",
            serialization_format="cloudpickle",
            code_paths=[str(ROOT / "src/telco_churn")],
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            pip_requirements=[
                "mlflow>=3,<4", "pandas>=2.2", "numpy>=1.26,<3",
                "scikit-learn>=1.5", "category-encoders>=2.6",
                "lightgbm>=4.3", "cloudpickle>=3",
            ],
        )

        version = mlflow.register_model(logged.model_uri, MODEL_NAME)
        MlflowClient().set_registered_model_alias(MODEL_NAME, "candidate", version=version.version)


if __name__ == "__main__":
    main()