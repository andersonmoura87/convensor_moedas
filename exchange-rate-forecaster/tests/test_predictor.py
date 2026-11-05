"""Unit tests for model loading and inference."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import src.serving.predictor as predictor
from src.features.engineering import FEATURE_COLS


def _scalers() -> tuple[MinMaxScaler, MinMaxScaler]:
    x_scaler = MinMaxScaler().fit(pd.DataFrame(np.zeros((2, 25)), columns=FEATURE_COLS))
    y_scaler = MinMaxScaler().fit(pd.DataFrame({"rate": [3.0, 5.0]}))
    return x_scaler, y_scaler


def test_load_model_uses_production_version_and_its_scalers() -> None:
    version = SimpleNamespace(version="4", run_id="run-4")
    client = MagicMock()
    client.get_latest_versions.return_value = [version]
    client.download_artifacts.return_value = "scalers.pkl"
    model = MagicMock()
    x_scaler, y_scaler = _scalers()

    with (
        patch.object(predictor, "_model", None),
        patch.object(predictor, "_x_scaler", None),
        patch.object(predictor, "_y_scaler", None),
        patch.object(predictor, "_sequence_length", None),
        patch.object(predictor.mlflow, "set_tracking_uri"),
        patch.object(predictor.mlflow, "MlflowClient", return_value=client),
        patch.object(
            predictor.mlflow.pytorch,
            "load_model",
            return_value=model,
        ) as load_model,
        patch.object(predictor, "_load_scalers", return_value=(x_scaler, y_scaler, 31)),
    ):
        predictor.load_model("http://mlflow:5000")

        assert predictor.model_loaded()
        assert predictor.expected_feature_shape() == (31, 25)

    load_model.assert_called_once_with(
        "models:/exchange-rate-lstm/4",
        map_location="cpu",
    )
    client.download_artifacts.assert_called_once_with("run-4", "scalers/scalers.pkl")


def test_predict_denormalizes_model_output() -> None:
    x_scaler, y_scaler = _scalers()
    model = MagicMock()
    model.return_value.numpy.return_value = np.array([[0.5]], dtype=np.float32)

    with (
        patch.object(predictor, "_model", model),
        patch.object(predictor, "_x_scaler", x_scaler),
        patch.object(predictor, "_y_scaler", y_scaler),
        patch.object(predictor, "_sequence_length", 30),
    ):
        result = predictor.predict(np.zeros((30, 25), dtype=np.float32))

    assert result == 4.0
