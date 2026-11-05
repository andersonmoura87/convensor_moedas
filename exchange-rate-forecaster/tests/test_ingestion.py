"""Unit tests for the ingestion layer (no network calls)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.fetch_rates import fetch_time_series, fetch_all_pairs, save_raw


MOCK_API_RESPONSE = {
    "base": "USD",
    "rates": {
        "2024-01-02": {"BRL": 4.85},
        "2024-01-03": {"BRL": 4.87},
        "2024-01-04": {"BRL": 4.82},
    },
}


@patch("src.ingestion.fetch_rates._get", return_value=MOCK_API_RESPONSE)
def test_fetch_time_series_returns_dataframe(mock_get: MagicMock) -> None:
    df = fetch_time_series("USD", "BRL", date(2024, 1, 2), date(2024, 1, 4))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["date", "base", "target", "rate"]
    assert len(df) == 3
    assert df["base"].unique() == ["USD"]
    assert df["target"].unique() == ["BRL"]


@patch("src.ingestion.fetch_rates._get", return_value=MOCK_API_RESPONSE)
def test_fetch_time_series_sorted_by_date(mock_get: MagicMock) -> None:
    df = fetch_time_series("USD", "BRL", date(2024, 1, 2), date(2024, 1, 4))
    assert list(df["date"]) == sorted(df["date"])


@patch("src.ingestion.fetch_rates._get", return_value={"base": "USD", "rates": {}})
def test_fetch_time_series_raises_on_empty_response(mock_get: MagicMock) -> None:
    with pytest.raises(ValueError, match="No data returned"):
        fetch_time_series("USD", "BRL", date(2024, 1, 2), date(2024, 1, 4))


@patch("src.ingestion.fetch_rates._get", return_value=MOCK_API_RESPONSE)
def test_fetch_all_pairs_returns_all_data(mock_get: MagicMock) -> None:
    df = fetch_all_pairs(pairs=[("USD", "BRL")])
    assert not df.empty
    assert "rate" in df.columns


def test_save_raw_creates_file(tmp_path: "pytest.FixtureRequest") -> None:
    import src.ingestion.fetch_rates as module
    original_raw_dir = module.RAW_DIR
    module.RAW_DIR = tmp_path  # type: ignore[assignment]

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "base": ["USD", "USD"],
        "target": ["BRL", "BRL"],
        "rate": [4.85, 4.87],
    })
    path = save_raw(df, filename="test_rates.csv")
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 2

    module.RAW_DIR = original_raw_dir
