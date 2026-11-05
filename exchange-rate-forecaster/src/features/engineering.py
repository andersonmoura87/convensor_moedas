"""
Feature engineering pipeline.

Transforms raw rate data into model-ready features:
  - Lag features (t-1 to t-N)
  - Rolling statistics (mean, std, min, max)
  - Calendar features (day of week, month, is_month_end)
  - Log-return (preferred over raw price for stationarity)
  - Normalisation using MinMaxScaler (fit on train, applied to all splits)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_raw(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["year"] = df["date"].dt.year
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3, 5, 7, 14, 21]
    df = df.copy()
    for lag in lags:
        df[f"rate_lag_{lag}"] = df.groupby(["base", "target"])["rate"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 30]
    df = df.copy()
    grouped = df.groupby(["base", "target"])["rate"]
    for w in windows:
        df[f"rate_roll_mean_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"rate_roll_std_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f"rate_roll_min_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).min())
        df[f"rate_roll_max_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).max())
    return df


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = df.groupby(["base", "target"])["rate"].transform(
        lambda x: np.log(x).diff()
    )
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_log_return(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("Feature matrix: %d rows × %d cols", *df.shape)
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — no data leakage."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    logger.info("Split sizes — train: %d | val: %d | test: %d", len(train), len(val), len(test))
    return train, val, test


FEATURE_COLS = [
    "day_of_week", "month", "quarter", "is_month_end", "year",
    "rate_lag_1", "rate_lag_2", "rate_lag_3", "rate_lag_5",
    "rate_lag_7", "rate_lag_14", "rate_lag_21",
    "rate_roll_mean_7", "rate_roll_std_7", "rate_roll_min_7", "rate_roll_max_7",
    "rate_roll_mean_14", "rate_roll_std_14", "rate_roll_min_14", "rate_roll_max_14",
    "rate_roll_mean_30", "rate_roll_std_30", "rate_roll_min_30", "rate_roll_max_30",
    "log_return",
]
TARGET_COL = "rate"


def fit_scalers(train: pd.DataFrame) -> tuple[MinMaxScaler, MinMaxScaler]:
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(train[FEATURE_COLS])
    y_scaler.fit(train[[TARGET_COL]])
    return x_scaler, y_scaler


def apply_scalers(
    df: pd.DataFrame,
    x_scaler: MinMaxScaler,
    y_scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    X = x_scaler.transform(df[FEATURE_COLS])
    y = y_scaler.transform(df[[TARGET_COL]]).ravel()
    return X, y


def run(raw_path: Path | str, pair: str = "USD/BRL") -> dict:
    """End-to-end feature pipeline for a single currency pair."""
    base, target = pair.split("/")
    df = load_raw(raw_path)
    df = df[(df["base"] == base) & (df["target"] == target)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No data for pair {pair} in {raw_path}")

    df = build_feature_matrix(df)
    train, val, test = train_test_split_temporal(df)
    x_scaler, y_scaler = fit_scalers(train)

    splits = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        X, y = apply_scalers(split, x_scaler, y_scaler)
        splits[name] = {"X": X, "y": y, "dates": split["date"].values}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "splits": splits,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_cols": FEATURE_COLS,
        "pair": pair,
    }
