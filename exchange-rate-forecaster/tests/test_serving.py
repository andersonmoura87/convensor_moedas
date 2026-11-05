"""Unit tests for the FastAPI serving layer."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import src.serving.predictor as predictor_module
from src.serving.main import PredictRequest, app, predict_endpoint


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_endpoint_when_model_not_loaded(client: TestClient) -> None:
    with patch.object(predictor_module, "_model", None):
        resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_endpoint_when_model_is_loaded(client: TestClient) -> None:
    with patch("src.serving.main.model_loaded", return_value=True):
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_predict_returns_503_when_model_not_loaded(client: TestClient) -> None:
    with patch.object(predictor_module, "_model", None):
        resp = client.post(
            "/predict",
            json={"pair": "USD/BRL", "features": [[0.5] * 25 for _ in range(30)]},
        )
    assert resp.status_code == 503


@pytest.mark.parametrize(
    "features",
    [
        [[0.5] * 25 for _ in range(29)],
        [[0.5] * 24 for _ in range(30)],
        [[0.5] * 25 for _ in range(29)] + [[0.5] * 24],
    ],
)
def test_predict_rejects_incompatible_feature_shape(
    client: TestClient,
    features: list[list[float]],
) -> None:
    resp = client.post("/predict", json={"pair": "USD/BRL", "features": features})
    assert resp.status_code == 422


def test_predict_rejects_non_finite_features(client: TestClient) -> None:
    features = [[0.5] * 25 for _ in range(30)]
    features[0][0] = float("nan")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(predict_endpoint(PredictRequest(pair="USD/BRL", features=features)))
    assert exc_info.value.status_code == 422


def test_predict_returns_prediction_when_model_loaded(client: TestClient) -> None:
    mock_rate = 4.95

    with (
        patch("src.serving.main.model_loaded", return_value=True),
        patch("src.serving.main.predict", return_value=mock_rate),
    ):
        resp = client.post(
            "/predict",
            json={"pair": "USD/BRL", "features": [[0.5] * 25 for _ in range(30)]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["pair"] == "USD/BRL"
    assert data["predicted_rate"] == pytest.approx(mock_rate)
    assert "latency_ms" in data


def test_predict_uses_loaded_model_sequence_length(client: TestClient) -> None:
    with (
        patch("src.serving.main.expected_feature_shape", return_value=(31, 25)),
        patch("src.serving.main.model_loaded", return_value=True),
        patch("src.serving.main.predict", return_value=4.95),
    ):
        resp = client.post(
            "/predict",
            json={"pair": "USD/BRL", "features": [[0.5] * 25 for _ in range(31)]},
        )

    assert resp.status_code == 200


def test_reload_endpoint_reports_success(client: TestClient) -> None:
    with patch("src.serving.main.load_model"):
        resp = client.post("/reload")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"


def test_reload_endpoint_reports_loading_failure(client: TestClient) -> None:
    with patch("src.serving.main.load_model", side_effect=RuntimeError("registry unavailable")):
        resp = client.post("/reload")
    assert resp.status_code == 500


def test_metrics_endpoint_returns_prometheus_format(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"prediction_requests_total" in resp.content
