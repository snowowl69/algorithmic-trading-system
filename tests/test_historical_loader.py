"""Tests for the Historical Loader — synthetic data generation (Indian markets)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engine.historical_loader import (
    generate_synthetic_bars,
    bars_to_models,
)


class TestSyntheticBars:
    def test_generates_data(self):
        # Use a time window that covers IST market hours (03:45-10:00 UTC)
        end = datetime(2026, 3, 6, 10, 0, tzinfo=timezone.utc)  # Friday
        start = end - timedelta(days=3)
        df = generate_synthetic_bars("RELIANCE", start=start, end=end)
        assert not df.empty
        assert "close" in df.columns
        assert "timestamp" in df.columns
        assert len(df) > 0

    def test_different_tickers_different_prices(self):
        end = datetime(2026, 3, 6, 10, 0, tzinfo=timezone.utc)
        start = end - timedelta(days=1)
        df_rel = generate_synthetic_bars("RELIANCE", start=start, end=end)
        df_tcs = generate_synthetic_bars("TCS", start=start, end=end)
        if not df_rel.empty and not df_tcs.empty:
            # Different base prices so opening values should differ
            assert abs(df_rel["close"].iloc[0] - df_tcs["close"].iloc[0]) > 1

    def test_ohlcv_sanity(self):
        end = datetime(2026, 3, 6, 10, 0, tzinfo=timezone.utc)
        start = end - timedelta(days=1)
        df = generate_synthetic_bars("INFY", start=start, end=end)
        if not df.empty:
            assert (df["high"] >= df["low"]).all()
            assert (df["volume"] > 0).all()

    def test_bars_to_models(self):
        end = datetime(2026, 3, 6, 10, 0, tzinfo=timezone.utc)
        start = end - timedelta(days=1)
        df = generate_synthetic_bars("RELIANCE", start=start, end=end)
        models = bars_to_models(df, "RELIANCE")
        assert len(models) == len(df)
        assert all(m.ticker == "RELIANCE" for m in models)
