"""
Model loader and inference wrapper.

Loads the latest Production model from the MLflow Model Registry
and the companion scalers, then exposes a `predict` function.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.features.engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

_model: Any = None
_x_scaler: MinMaxScaler | None = None
_y_scaler: MinMaxScaler | None = None
_sequence_length: int | None = None

MODEL_NAME = "exchange-rate-lstm"
MODEL_STAGE = "Production"
DEFAULT_SEQUENCE_LENGTH = 30


def _load_scalers(artifact_path: str) -> tuple[MinMaxScaler, MinMaxScaler, int]:
    with open(artifact_path, "rb") as f:
        data = pickle.load(f)
    return data["x_scaler"], data["y_scaler"], data.get("seq_len", DEFAULT_SEQUENCE_LENGTH)


def expected_feature_shape() -> tuple[int, int]:
    return _sequence_length or DEFAULT_SEQUENCE_LENGTH, len(FEATURE_COLS)


def load_model(tracking_uri: str = "http://mlflow:5000") -> None:
    """Loads model from registry. Called once at app startup."""
    global _model, _x_scaler, _y_scaler, _sequence_length

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(f"No {MODEL_STAGE} model version found for {MODEL_NAME}.")

    version = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{version.version}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    model.eval()

    scaler_path = client.download_artifacts(version.run_id, "scalers/scalers.pkl")
    x_scaler, y_scaler, sequence_length = _load_scalers(scaler_path)
    if sequence_length <= 0:
        raise RuntimeError(f"Invalid model sequence length: {sequence_length}.")
    if len(x_scaler.feature_names_in_) != len(FEATURE_COLS):
        raise RuntimeError(
            f"Scaler expects {len(x_scaler.feature_names_in_)} features; "
            f"serving expects {len(FEATURE_COLS)}."
        )

    _model, _x_scaler, _y_scaler, _sequence_length = (
        model,
        x_scaler,
        y_scaler,
        sequence_length,
    )
    logger.info("Model and scalers loaded successfully")


def predict(features: np.ndarray) -> float:
    """
    Runs inference on a (seq_len, n_features) array.

    Returns:
        Predicted exchange rate (denormalised).
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    expected_shape = expected_feature_shape()
    if features.shape != expected_shape:
        raise ValueError(f"Expected features with shape {expected_shape}, got {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("Features must contain only finite numbers.")
    if _x_scaler is None or _y_scaler is None:
        raise RuntimeError("Model scalers not loaded.")

    x_scaled = _x_scaler.transform(features)
    tensor = torch.from_numpy(x_scaled.astype(np.float32)).unsqueeze(0)  # (1, seq_len, n_features)

    with torch.no_grad():
        pred_scaled = _model(tensor).numpy()

    rate = float(_y_scaler.inverse_transform(pred_scaled)[0, 0])
    return rate


def model_loaded() -> bool:
    return (
        _model is not None
        and _x_scaler is not None
        and _y_scaler is not None
        and _sequence_length is not None
    )
