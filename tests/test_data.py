import pandas as pd
import pytest

from telco_churn.data import load_raw_data, preprocess_data, extract_target, make_holdout_split


def test_load_raw_data_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "missing.csv")


def test_preprocess_data_cleans_columns_and_fills_numeric_values():
    raw = pd.DataFrame({
        " customerID ": ["001", "002"],
        "CustomerID": ["001", "002"],
        "customer_id": ["001", "002"],
        " TotalCharges ": ["45.5", " "],
        "SeniorCitizen": [None, 1],
        "tenure": [None, 5],
        "Partner": ["Yes", None],
    })

    cleaned = preprocess_data(raw)
    assert cleaned.columns.tolist() == ["TotalCharges", "SeniorCitizen", "tenure", "Partner"]
    assert cleaned["TotalCharges"].tolist() == [45.5, 0.0]
    assert cleaned["SeniorCitizen"].tolist() == [0, 1]
    assert pd.api.types.is_integer_dtype(cleaned["SeniorCitizen"])
    assert cleaned["tenure"].tolist() == [0.0, 5.0]
    assert cleaned.loc[0, "Partner"] == "Yes"
    assert pd.isna(cleaned.loc[1, "Partner"])


def test_extract_target_maps_labels_and_removes_target_column():
    X, y = extract_target(pd.DataFrame({"Outcome": ["Yes", "No"], "tenure": [1, 2]}), "Outcome")
    assert "Outcome" not in X.columns
    assert y.tolist() == [1, 0]


def test_make_holdout_split_is_disjoint_and_stratified():
    X = pd.DataFrame({"row": range(100)})
    y = pd.Series([0] * 70 + [1] * 30)

    X_train, X_test, y_train, y_test = make_holdout_split(X, y, 0.2, 42)

    assert set(X_train.index).isdisjoint(X_test.index)
    assert y_train.value_counts().to_dict() == {0: 56, 1: 24}
    assert y_test.value_counts().to_dict() == {0: 14, 1: 6}