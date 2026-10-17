import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import download
import train
from api.main import FEATURE_COLUMNS, MODEL_PATH, app
from download import clean_data
from train import build_pipeline

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifact():
    """Ensure the trained model artifact exists before API tests run."""
    if not MODEL_PATH.exists():
        if not train.DATA_PATH.exists():
            download.main()
        train.main()


def test_clean_data_maps_binary_target_and_drops_missing_target():
    raw_sample = pd.DataFrame(
        {
            "age": [63.0, 67.0, 58.0, 50.0],
            "ca": [0.0, float("nan"), 2.0, 1.0],
            "target_raw": [0.0, 2.0, 4.0, float("nan")],
        }
    )
    cleaned, removed = clean_data(raw_sample)

    assert removed == 1
    assert len(cleaned) == 3
    assert "target_raw" not in cleaned.columns
    assert list(cleaned["target"]) == [0, 1, 1]
    assert cleaned["ca"].isna().sum() == 1


def test_model_pipeline_imputes_missing_values_and_predicts_probabilities():
    sample_features = pd.DataFrame(
        [
            [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0.0, 6.0],
            [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, float("nan"), 3.0],
            [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0.0, float("nan")],
            [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2.0, 3.0],
        ],
        columns=FEATURE_COLUMNS,
    )
    sample_target = pd.Series([0, 1, 0, 1])

    pipeline = build_pipeline(
        LogisticRegression(max_iter=500, random_state=42)
    )
    pipeline.fit(sample_features, sample_target)

    predictions = pipeline.predict(sample_features)
    probabilities = pipeline.predict_proba(sample_features)[:, 1]

    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})
    assert all(0.0 <= prob <= 1.0 for prob in probabilities)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_requests"] >= 1
    assert "uptime_seconds" in data
    assert "predictions_generated" in data
    assert "validation_errors_422" in data


def test_predict_endpoint():
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["label"] in {
        "Heart disease predicted",
        "No heart disease predicted",
    }


def test_predict_logs_request_without_patient_values(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    request_log = (
        "API request completed: method=POST "
        "path=/predict status_code=200"
    )
    assert request_log in caplog.text
    assert (
        "Prediction generated for endpoint=/predict; input_values_logged=false"
        in caplog.text
    )
    assert "age" not in caplog.text
    assert "chol" not in caplog.text
    assert "212" not in caplog.text


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("cp", 0),
        ("cp", 1.5),
        ("thal", 4),
        ("age", 100),
    ],
)
def test_predict_rejects_values_outside_training_data(field, invalid_value):
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }
    payload[field] = invalid_value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_target_field():
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
        "target": 0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422