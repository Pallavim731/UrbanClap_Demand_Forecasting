import os

import joblib
import pandas as pd


BASE_DIR = ".."


def test_original_dataset_exists():
    path = os.path.join(BASE_DIR, "data", "clickstream.csv")

    assert os.path.exists(path)


def test_original_dataset_shape():
    path = os.path.join(BASE_DIR, "data", "clickstream.csv")

    df = pd.read_csv(path)

    assert df.shape[0] == 1000
    assert df.shape[1] == 16


def test_processed_training_data():
    path = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")

    df = pd.read_csv(path)

    assert df.shape[0] == 800
    assert df.shape[1] > 0


def test_processed_testing_data():
    path = os.path.join(BASE_DIR, "data", "processed", "X_test.csv")

    df = pd.read_csv(path)

    assert df.shape[0] == 200
    assert df.shape[1] > 0


def test_target_values():
    path = os.path.join(BASE_DIR, "data", "processed", "y_train.csv")

    df = pd.read_csv(path)

    assert set(df["label_purchased"].unique()).issubset({0, 1})


def test_model_exists():
    path = os.path.join(
        BASE_DIR,
        "models",
        "baseline_logistic_regression.pkl",
    )

    assert os.path.exists(path)


def test_model_prediction():
    model_path = os.path.join(
        BASE_DIR,
        "models",
        "baseline_logistic_regression.pkl",
    )

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "X_test.csv",
    )

    model = joblib.load(model_path)
    X_test = pd.read_csv(data_path)

    predictions = model.predict(X_test)

    assert len(predictions) == 200
    assert set(predictions).issubset({0, 1})