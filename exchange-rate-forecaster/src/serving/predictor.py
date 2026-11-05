"""
Model loader and inference wrapper.

Loads the latest Production model from the MLflow Model Registry
and the companion scalers, then exposes a `predict` function.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow.pytorch
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

_model: Any = None
_x_scaler: MinMaxScaler | None = None
_y_scaler: MinMaxScaler | None = None

SCALERS_PATH = Path(__file__).resolve().parents[2] / "mlflow" / "artifacts" / "scalers.pkl"
MODEL_NAME = "exchange-rate-lstm"
MODEL_STAGE = "Production"


def _load_scalers() -> tuple[MinMaxScaler, MinMaxScaler]:
    with open(SCALERS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["x_scaler"], data["y_scaler"]


def load_model(tracking_uri: str = "http://mlflow:5000") -> None:
    """Loads model from registry. Called once at app startup."""
    global _model, _x_scaler, _y_scaler

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from %s", model_uri)
    _model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    _model.eval()

    _x_scaler, _y_scaler = _load_scalers()
    logger.info("Model and scalers loaded successfully")


def predict(features: np.ndarray) -> float:
    """
    Runs inference on a (seq_len, n_features) array.

    Returns:
        Predicted exchange rate (denormalised).
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    x_scaled = _x_scaler.transform(features)
    tensor = torch.from_numpy(x_scaled.astype(np.float32)).unsqueeze(0)  # (1, seq_len, n_features)

    with torch.no_grad():
        pred_scaled = _model(tensor).numpy()

    rate = float(_y_scaler.inverse_transform(pred_scaled)[0, 0])
    return rate


def model_loaded() -> bool:
    return _model is not None
