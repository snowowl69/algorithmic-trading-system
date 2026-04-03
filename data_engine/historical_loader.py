"""
Historical OHLCV data loader for Indian markets (NSE / BSE / MCX).

Primary provider: **yfinance** (free, no API keys needed).
Fallback: synthetic GBM generator with realistic INR prices.

Used by the backtesting engine and by strategies that need a lookback buffer
before switching to live streaming.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import SYSTEM
from core.models import Bar

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  yfinance — NSE / BSE historical bars
# ═══════════════════════════════════════════════════════════════════════

# Map our tickers to yfinance symbols (NSE suffix .NS)
_YFINANCE_MAP = {
    # NSE equities
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS", "BHARTIARTL": "BHARTIARTL.NS", "ITC": "ITC.NS",
    "HINDUNILVR": "HINDUNILVR.NS", "WIPRO": "WIPRO.NS",
    "LT": "LT.NS", "AXISBANK": "AXISBANK.NS", "KOTAKBANK": "KOTAKBANK.NS",
    "MARUTI": "MARUTI.NS", "TATAMOTORS": "TATAMOTORS.NS",
    "ADANIENT": "ADANIENT.NS", "BAJFINANCE": "BAJFINANCE.NS",
    "SUNPHARMA": "SUNPHARMA.NS", "TITAN": "TITAN.NS",
    "NESTLEIND": "NESTLEIND.NS",
    # MCX commodities — use MCX India futures via yfinance
    "GOLD": "GC=F",       # Gold futures (USD proxy — converted to INR in synthetic)
    "SILVER": "SI=F",     # Silver futures
    "CRUDEOIL": "CL=F",   # Crude oil futures
    "NATURALGAS": "NG=F", # Natural gas futures
    "COPPER": "HG=F",     # Copper futures
}


def fetch_yfinance_bars(
    ticker: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Fetch historical bars from yfinance for NSE/BSE/MCX tickers.
    Returns a DataFrame with: timestamp, open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — falling back to synthetic data")
        return pd.DataFrame()

    start = start or (datetime.now(timezone.utc) - timedelta(days=5))
    end = end or datetime.now(timezone.utc)

    yf_symbol = _YFINANCE_MAP.get(ticker, f"{ticker}.NS")

    # yfinance limits: 1m data is only available for last 7 days
    # For longer periods, use daily bars
    day_diff = (end - start).days
    if day_diff > 7 and interval in ("1m", "2m", "5m"):
        interval = "1d"  # switch to daily for longer periods

    try:
        tk_obj = yf.Ticker(yf_symbol)
        df = tk_obj.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
        )

        if df.empty:
            log.warning("No yfinance data for %s (%s)", ticker, yf_symbol)
            return pd.DataFrame()

        df = df.reset_index()
        # Normalise column names
        col_map = {
            "Date": "timestamp", "Datetime": "timestamp",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        df.rename(columns=col_map, inplace=True)

        if "timestamp" not in df.columns:
            # Some yfinance versions use different index name
            df["timestamp"] = df.index

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        keep = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        log.info("Fetched %d bars for %s (%s) via yfinance", len(df), ticker, yf_symbol)
        return df

    except Exception as exc:
        log.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
#  Convenience helpers
# ═══════════════════════════════════════════════════════════════════════

def bars_to_models(df: pd.DataFrame, ticker: str) -> List[Bar]:
    """Convert a DataFrame of bars into a list of ``Bar`` model objects."""
    out: List[Bar] = []
    for _, row in df.iterrows():
        out.append(
            Bar(
                ticker=ticker,
                timestamp=row["timestamp"].to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                vwap=row.get("vwap"),
            )
        )
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Synthetic data generator — Indian stocks (INR prices)
# ═══════════════════════════════════════════════════════════════════════

# Realistic base prices in INR (approximate current market prices)
_INR_BASE_PRICES = {
    # NSE equities
    "RELIANCE": 2950.0,   "TCS": 4200.0,       "INFY": 1850.0,
    "HDFCBANK": 1650.0,   "ICICIBANK": 1250.0,  "SBIN": 820.0,
    "BHARTIARTL": 1680.0, "ITC": 470.0,         "HINDUNILVR": 2450.0,
    "WIPRO": 560.0,       "LT": 3600.0,         "AXISBANK": 1180.0,
    "KOTAKBANK": 1900.0,  "MARUTI": 12500.0,    "TATAMOTORS": 780.0,
    "ADANIENT": 3200.0,   "BAJFINANCE": 7400.0, "SUNPHARMA": 1750.0,
    "TITAN": 3350.0,      "NESTLEIND": 2350.0,
    # MCX commodities (INR per unit)
    "GOLD": 72500.0,      "SILVER": 88000.0,
    "CRUDEOIL": 6800.0,   "NATURALGAS": 250.0,  "COPPER": 820.0,
}


def generate_synthetic_bars(
    ticker: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval_minutes: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV bars for Indian market demo / backtesting.
    Uses geometric Brownian motion with INR base prices.
    Market hours: IST 09:15 – 15:30 (UTC 03:45 – 10:00).
    """
    start = start or (datetime.now(timezone.utc) - timedelta(days=5))
    end = end or datetime.now(timezone.utc)

    timestamps = pd.date_range(start=start, end=end, freq=f"{interval_minutes}min", tz="UTC")
    # Filter to weekday IST market-hours only (09:15-15:30 IST = 03:45-10:00 UTC)
    timestamps = timestamps[timestamps.weekday < 5]
    timestamps = timestamps[
        (timestamps.hour * 60 + timestamps.minute >= 3 * 60 + 45) &
        (timestamps.hour * 60 + timestamps.minute < 10 * 60)
    ]

    if len(timestamps) == 0:
        log.warning("No timestamps generated for %s", ticker)
        return pd.DataFrame()

    n = len(timestamps)
    rng = np.random.default_rng(seed=hash(ticker) & 0xFFFFFFFF)

    base = _INR_BASE_PRICES.get(ticker, 1000.0 + rng.random() * 3000)

    # Geometric Brownian Motion
    dt = 1 / (252 * 375)  # fraction of a trading year per minute (375 min/day for NSE)
    mu = 0.12             # annualised drift (Indian market tends higher)
    sigma = 0.22          # annualised volatility
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = base * np.exp(np.cumsum(log_returns))

    spread = close * 0.002 * (1 + rng.random(n))  # ~0.2-0.4% candle spread
    high = close + spread * rng.random(n)
    low = close - spread * rng.random(n)
    opn = low + (high - low) * rng.random(n)
    volume = (rng.exponential(scale=15_000, size=n) + 500).astype(float)

    df = pd.DataFrame({
        "timestamp": timestamps[:n],
        "open": np.round(opn, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": np.round(volume, 0),
    })
    log.info("Generated %d synthetic bars for %s (INR)", len(df), ticker)
    return df


def fetch_bars(
    ticker: str,
    provider: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Unified entry point — dispatches to yfinance, falls back to synthetic.
    """
    provider = (provider or SYSTEM.data_provider).lower()

    if provider == "yfinance":
        try:
            df = fetch_yfinance_bars(ticker, **kwargs)
            if not df.empty:
                return df
        except Exception as exc:
            log.warning("yfinance failed for %s (%s) — falling back to synthetic", ticker, exc)

    log.info("Using synthetic data for %s", ticker)
    return generate_synthetic_bars(ticker, **kwargs)
