"""Unit tests for model training components."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from src.features.engineering import FEATURE_COLS
from src.training.dataset import RateSequenceDataset
from src.training.model import build_model
from src.training.train import train


def test_sequence_dataset_builds_sliding_windows() -> None:
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.arange(20, dtype=np.float32)
    dataset = RateSequenceDataset(X, y, seq_len=5)

    features, target = dataset[0]

    assert len(dataset) == 15
    assert features.shape == (5, 3)
    assert target.item() == pytest.approx(5)


def test_sequence_dataset_rejects_too_short_data() -> None:
    with pytest.raises(ValueError, match="Dataset too short"):
        RateSequenceDataset(np.zeros((5, 2)), np.zeros(5), seq_len=5)


def test_model_returns_one_prediction_per_sequence() -> None:
    model = build_model(
        input_size=25,
        config={"hidden_size": 4, "num_layers": 1, "dropout": 0.0, "fc_hidden": 2},
    )

    predictions = model(torch.zeros((3, 10, 25)))

    assert predictions.shape == (3, 1)


def _feature_artifact() -> dict:
    rng = np.random.default_rng(42)
    splits = {}
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    train_X = rng.random((60, len(FEATURE_COLS)))
    train_y = rng.random(60)
    x_scaler.fit(pd.DataFrame(train_X, columns=FEATURE_COLS))
    y_scaler.fit(pd.DataFrame({"rate": train_y}))

    for split_name, row_count in (("train", 60), ("val", 30), ("test", 30)):
        X = rng.random((row_count, len(FEATURE_COLS)))
        y = rng.random(row_count)
        splits[split_name] = {"X": X, "y": y}

    return {
        "pair": "USD/BRL",
        "splits": splits,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def test_train_consumes_features_and_writes_metrics(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.pkl"
    metrics_path = tmp_path / "metrics.json"
    with feature_path.open("wb") as feature_file:
        pickle.dump(_feature_artifact(), feature_file)

    run_context = MagicMock()
    run_context.__enter__.return_value.info.run_id = "test-run"
    mlflow_client = MagicMock()

    with (
        patch("src.training.train.mlflow.set_experiment"),
        patch("src.training.train.mlflow.start_run", return_value=run_context),
        patch("src.training.train.mlflow.log_params"),
        patch("src.training.train.mlflow.log_param"),
        patch("src.training.train.mlflow.log_metrics"),
        patch("src.training.train.mlflow.log_artifact"),
        patch("src.training.train.mlflow.models.infer_signature"),
        patch(
            "src.training.train.mlflow.pytorch.log_model",
            return_value=SimpleNamespace(registered_model_version=2),
        ),
        patch("src.training.train.mlflow.MlflowClient", return_value=mlflow_client),
    ):
        run_id = train(
            {
                "features_path": feature_path,
                "metrics_path": metrics_path,
                "seq_len": 5,
                "epochs": 1,
                "batch_size": 16,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "fc_hidden": 2,
                "patience": 1,
            }
        )

    assert run_id == "test-run"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "test_mae" in metrics
    mlflow_client.transition_model_version_stage.assert_called_once_with(
        name="exchange-rate-lstm",
        version=2,
        stage="Production",
        archive_existing_versions=True,
    )


def test_train_rejects_features_for_a_different_pair(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.pkl"
    with feature_path.open("wb") as feature_file:
        pickle.dump({"pair": "EUR/USD"}, feature_file)

    with pytest.raises(ValueError, match="does not match training pair"):
        train({"features_path": feature_path})
