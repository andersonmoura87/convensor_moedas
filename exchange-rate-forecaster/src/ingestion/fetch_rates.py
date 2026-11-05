"""
Fetches historical exchange rate data from the Frankfurter API
and persists raw CSV files under data/raw/.

Frankfurter is a free, open-source API backed by the ECB dataset,
no key required. Rates are available from 1999-01-04.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://api.frankfurter.dev/v1"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SUPPORTED_PAIRS = [
    ("USD", "BRL"),
    ("EUR", "BRL"),
    ("EUR", "USD"),
    ("GBP", "USD"),
    ("JPY", "USD"),
]


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_time_series(
    base: str,
    target: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Returns a tidy DataFrame with columns [date, base, target, rate]."""
    # Frankfurter v1 usa ".." entre datas no path, sem parâmetros from/to
    params = {"base": base, "symbols": target}
    data = _get(f"{BASE_URL}/{start}..{end}", params)
    rates = data.get("rates", {})

    records = [
        {"date": pd.Timestamp(d), "base": base, "target": target, "rate": v[target]}
        for d, v in rates.items()
    ]
    if not records:
        raise ValueError(f"No data returned for {base}/{target} [{start} – {end}]")

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    logger.info("Fetched %d rows for %s/%s", len(df), base, target)
    return df


def fetch_all_pairs(
    start: date | None = None,
    end: date | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    end = end or date.today() - timedelta(days=1)
    start = start or end - timedelta(days=5 * 365)
    pairs = pairs or SUPPORTED_PAIRS

    frames = []
    for base, target in pairs:
        try:
            df = fetch_time_series(base, target, start, end)
            frames.append(df)
        except Exception:
            logger.exception("Failed to fetch %s/%s", base, target)

    if not frames:
        raise RuntimeError("No data could be retrieved for any pair.")

    return pd.concat(frames, ignore_index=True)


def save_raw(df: pd.DataFrame, filename: str = "rates_raw.csv") -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def run(
    start: str | None = None,
    end: str | None = None,
    output: str = "rates_raw.csv",
) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_d = date.fromisoformat(start) if start else None
    end_d = date.fromisoformat(end) if end else None
    df = fetch_all_pairs(start=start_d, end=end_d)
    return save_raw(df, filename=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical exchange rates")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="rates_raw.csv")
    args = parser.parse_args()
    path = run(start=args.start, end=args.end, output=args.output)
    print(f"Data saved to {path}")
