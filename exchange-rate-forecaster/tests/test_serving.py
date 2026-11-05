"""Unit tests for the FastAPI serving layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import src.serving.predictor as predictor_module
from src.serving.main import app


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


def test_predict_returns_503_when_model_not_loaded(client: TestClient) -> None:
    with patch.object(predictor_module, "_model", None):
        resp = client.post(
            "/predict",
            json={"pair": "USD/BRL", "features": [[0.5] * 25]},
        )
    assert resp.status_code == 503


def test_predict_returns_prediction_when_model_loaded(client: TestClient) -> None:
    mock_rate = 4.95

    with (
        patch("src.serving.main.model_loaded", return_value=True),
        patch("src.serving.main.predict", return_value=mock_rate),
    ):
        resp = client.post(
            "/predict",
            json={"pair": "USD/BRL", "features": [[0.5] * 25]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["pair"] == "USD/BRL"
    assert data["predicted_rate"] == pytest.approx(mock_rate)
    assert "latency_ms" in data


def test_metrics_endpoint_returns_prometheus_format(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"prediction_requests_total" in resp.content
