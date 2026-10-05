import json

import pytest

from milestone_4.api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "running"
    assert data["message"] == "UrbanClap ML Prediction API"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    features = [0.0] * 53

    response = client.post(
        "/predict",
        data=json.dumps({"features": features}),
        content_type="application/json",
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data
    assert len(data["prediction"]) == 1


def test_predict_missing_features(client):
    response = client.post(
        "/predict",
        data=json.dumps({}),
        content_type="application/json",
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


def test_predict_invalid_features(client):
    features = [0.0] * 10

    response = client.post(
        "/predict",
        data=json.dumps({"features": features}),
        content_type="application/json",
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Prediction failed"