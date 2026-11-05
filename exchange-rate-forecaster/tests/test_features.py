"""Unit tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_calendar_features,
    add_lag_features,
    add_log_return,
    add_rolling_features,
    build_feature_matrix,
    train_test_split_temporal,
    fit_scalers,
    apply_scalers,
    FEATURE_COLS,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal tidy rate DataFrame for 90 days."""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "base": "USD",
        "target": "BRL",
        "rate": 4.8 + rng.normal(0, 0.05, size=90).cumsum(),
    })


def test_add_calendar_features_columns(sample_df: pd.DataFrame) -> None:
    df = add_calendar_features(sample_df)
    for col in ("day_of_week", "month", "quarter", "is_month_end", "year"):
        assert col in df.columns


def test_add_lag_features_no_data_leakage(sample_df: pd.DataFrame) -> None:
    df = add_lag_features(sample_df, lags=[1, 7])
    # lag_1 for row i should equal rate at row i-1
    assert df.iloc[5]["rate_lag_1"] == pytest.approx(df.iloc[4]["rate"])


def test_add_rolling_features_shape(sample_df: pd.DataFrame) -> None:
    df = add_rolling_features(sample_df, windows=[7])
    assert "rate_roll_mean_7" in df.columns
    assert len(df) == len(sample_df)


def test_add_log_return_first_value_nan(sample_df: pd.DataFrame) -> None:
    df = add_log_return(sample_df)
    assert pd.isna(df.iloc[0]["log_return"])
    assert not pd.isna(df.iloc[1]["log_return"])


def test_build_feature_matrix_no_nans(sample_df: pd.DataFrame) -> None:
    df = build_feature_matrix(sample_df)
    assert df.isna().sum().sum() == 0


def test_train_test_split_no_overlap(sample_df: pd.DataFrame) -> None:
    df = build_feature_matrix(sample_df)
    train, val, test = train_test_split_temporal(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["date"].max() <= val["date"].min()
    assert val["date"].max() <= test["date"].min()


def test_scalers_roundtrip(sample_df: pd.DataFrame) -> None:
    df = build_feature_matrix(sample_df)
    train, val, _ = train_test_split_temporal(df)
    x_scaler, y_scaler = fit_scalers(train)
    X_train, y_train = apply_scalers(train, x_scaler, y_scaler)
    assert X_train.shape[1] == len(FEATURE_COLS)
    assert X_train.min() >= -0.01  # scaled values roughly in [0, 1]
    assert X_train.max() <= 1.01
