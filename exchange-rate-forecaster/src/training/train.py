"""
Training pipeline with MLflow experiment tracking.

Responsibilities:
  1. Load raw data and run feature engineering
  2. Build PyTorch DataLoaders
  3. Train the LSTM model
  4. Evaluate on val/test sets (MAE, RMSE, MAPE)
  5. Log params, metrics and artefacts to MLflow
  6. Register the best model in the MLflow Model Registry
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.features.engineering import FEATURE_COLS, run as build_features
from src.training.dataset import RateSequenceDataset
from src.training.model import build_model

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "pair": "USD/BRL",
    "seq_len": 30,
    "batch_size": 64,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "fc_hidden": 64,
    "patience": 10,
    "grad_clip": 1.0,
}

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "mlflow" / "artifacts"


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {
        f"{label}_mae": mae,
        f"{label}_rmse": rmse,
        f"{label}_mape": mape,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds.append(model(X_batch).cpu().numpy())
    return np.concatenate(preds).ravel()


def train(config: dict | None = None) -> str:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "rates_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run ingestion first.")

    logger.info("Building features for pair %s", cfg["pair"])
    feature_data = build_features(raw_path, pair=cfg["pair"])
    splits = feature_data["splits"]
    y_scaler = feature_data["y_scaler"]
    x_scaler = feature_data["x_scaler"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        ds = RateSequenceDataset(
            splits[split_name]["X"],
            splits[split_name]["y"],
            seq_len=cfg["seq_len"],
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=(split_name == "train"),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    model = build_model(len(FEATURE_COLS), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    mlflow.set_experiment("exchange-rate-lstm")

    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("device", str(device))

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict = {}

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            train_losses = []
            for X_batch, y_batch in loaders["train"]:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in loaders["val"]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_losses.append(criterion(model(X_batch), y_batch).item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            scheduler.step(val_loss)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            logger.info("Epoch %3d | train_loss=%.6f | val_loss=%.6f", epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        model.load_state_dict(best_state)

        # Evaluation on val and test sets
        for split_name in ("val", "test"):
            raw_preds_scaled = evaluate(model, loaders[split_name], device)
            raw_preds = y_scaler.inverse_transform(raw_preds_scaled.reshape(-1, 1)).ravel()
            raw_true_scaled = splits[split_name]["y"][cfg["seq_len"]:]
            raw_true = y_scaler.inverse_transform(raw_true_scaled.reshape(-1, 1)).ravel()
            metrics = _regression_metrics(raw_true, raw_preds, split_name)
            mlflow.log_metrics(metrics)
            logger.info("%s metrics: %s", split_name, metrics)

        # Save scalers as artefacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        scaler_path = MODELS_DIR / "scalers.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)
        mlflow.log_artifact(str(scaler_path), artifact_path="scalers")

        # Log model to registry
        signature = mlflow.models.infer_signature(
            splits["train"]["X"][:1],
            np.array([[0.0]]),
        )
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="exchange-rate-lstm",
        )

        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="USD/BRL")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    run_id = train({
        "pair": args.pair,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "seq_len": args.seq_len,
    })
    print(f"Training complete. Run ID: {run_id}")
