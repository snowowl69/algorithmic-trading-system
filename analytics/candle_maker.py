"""
Professional Candlestick Making & Order Flow Engine.

Designed from 30+ years of institutional trading desk practices:
1. Real-Time Candle Making: Aggregates sub-second ticks into clean OHLCV bars
   across 1m, 3m, 5m, 15m, 1h timeframes.
2. Order Flow Volume Delta: Tracks Buy Volume, Sell Volume, and Cumulative
   Volume Delta (CVD) to identify aggressive market orders vs passive limit orders.
3. Heikin-Ashi Smoothing: True recursive Heikin-Ashi calculation to filter noise.
4. Visible Range Volume Profile (VRVP): Calculates Point of Control (POC),
   Value Area High (VAH), and Value Area Low (VAL) for auction market analysis.
5. Anchored VWAP with Multi-Sigma Bands: Computes intraday volume-weighted average
   price with +/-1 and +/-2 standard deviation volatility envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LiveTick:
    """Sub-second price & volume tick."""
    ticker: str
    price: float
    volume: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_buyer_maker: bool = False  # False = aggressive buy, True = aggressive sell


@dataclass
class FormingCandle:
    """The currently open, active candle that is being constructed in real-time."""
    ticker: str
    timeframe: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    tick_count: int = 0

    @property
    def delta(self) -> float:
        """Net volume delta: aggressive buyer volume minus aggressive seller volume."""
        return self.buy_volume - self.sell_volume

    @property
    def range(self) -> float:
        return self.high - self.low


class LiveCandleMaker:
    """
    High-performance candle synthesis and volume profiling engine.
    Maintains completed bars and forming candles per ticker.
    """

    def __init__(self, bar_history_limit: int = 500) -> None:
        self.bar_history_limit = bar_history_limit
        self._forming_candles: Dict[str, FormingCandle] = {}
        self._completed_bars: Dict[str, List[Dict[str, Any]]] = {}
        self._cvd_history: Dict[str, float] = {}

    def ingest_tick(self, tick: LiveTick, timeframe_sec: int = 60) -> Optional[Dict[str, Any]]:
        """
        Ingest a live tick, update the forming candle, and complete the bar
        when the timeframe boundary has been crossed.
        """
        ticker = tick.ticker
        ts = tick.timestamp
        px = tick.price
        vol = tick.volume

        # Determine interval start timestamp
        epoch_sec = int(ts.timestamp())
        interval_start_epoch = (epoch_sec // timeframe_sec) * timeframe_sec
        interval_start = datetime.fromtimestamp(interval_start_epoch, tz=timezone.utc)

        forming = self._forming_candles.get(ticker)
        completed_bar = None

        if forming is None or forming.open_time < interval_start:
            # Complete the previous candle if it exists
            if forming is not None:
                completed_bar = {
                    "timestamp": forming.open_time.isoformat(),
                    "open": forming.open,
                    "high": forming.high,
                    "low": forming.low,
                    "close": forming.close,
                    "volume": forming.volume,
                    "buy_volume": forming.buy_volume,
                    "sell_volume": forming.sell_volume,
                    "delta": forming.delta,
                    "tick_count": forming.tick_count,
                }
                if ticker not in self._completed_bars:
                    self._completed_bars[ticker] = []
                self._completed_bars[ticker].append(completed_bar)
                if len(self._completed_bars[ticker]) > self.bar_history_limit:
                    self._completed_bars[ticker].pop(0)

                # Update cumulative volume delta
                self._cvd_history[ticker] = self._cvd_history.get(ticker, 0.0) + forming.delta

            # Start new forming candle
            is_buy = not tick.is_buyer_maker
            forming = FormingCandle(
                ticker=ticker,
                timeframe=f"{timeframe_sec}s",
                open_time=interval_start,
                open=px,
                high=px,
                low=px,
                close=px,
                volume=vol,
                buy_volume=vol if is_buy else 0.0,
                sell_volume=0.0 if is_buy else vol,
                tick_count=1,
            )
            self._forming_candles[ticker] = forming
        else:
            # Update existing forming candle
            forming.high = max(forming.high, px)
            forming.low = min(forming.low, px)
            forming.close = px
            forming.volume += vol
            forming.tick_count += 1
            if tick.is_buyer_maker:
                forming.sell_volume += vol
            else:
                forming.buy_volume += vol

        return completed_bar

    def get_forming_candle(self, ticker: str) -> Optional[FormingCandle]:
        """Return the current forming candle for a ticker."""
        return self._forming_candles.get(ticker)

    def get_bars_dataframe(self, ticker: str) -> pd.DataFrame:
        """Return completed bars as a pandas DataFrame."""
        bars = self._completed_bars.get(ticker, [])
        if not bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "delta"])
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        return df


# ═══════════════════════════════════════════════════════════════════════
#  Order Flow: Volume Profile (VRVP) & Point of Control (POC)
# ═══════════════════════════════════════════════════════════════════════

def compute_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 40,
    value_area_pct: float = 0.70,
) -> Dict[str, Any]:
    """
    Computes the Volume Profile across the visible price range:
    - POC (Point of Control): Strike/price with the heaviest traded volume
    - VAH (Value Area High): Upper price bound enclosing 70% of volume
    - VAL (Value Area Low): Lower price bound enclosing 70% of volume
    """
    if df.empty or "close" not in df.columns or "volume" not in df.columns:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "bins": [], "volumes": []}

    high_max = float(df["high"].max() if "high" in df.columns else df["close"].max())
    low_min = float(df["low"].min() if "low" in df.columns else df["close"].min())

    if high_max <= low_min:
        return {"poc": high_max, "vah": high_max, "val": low_min, "bins": [high_max], "volumes": [1.0]}

    bins = np.linspace(low_min, high_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    volumes = np.zeros(num_bins, dtype=float)

    closes = df["close"].values
    vols = df["volume"].values

    # Distribute bar volume to the corresponding price bin
    bin_indices = np.digitize(closes, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    for idx, v in zip(bin_indices, vols):
        volumes[idx] += v

    total_volume = np.sum(volumes)
    if total_volume == 0:
        return {"poc": high_max, "vah": high_max, "val": low_min, "bins": bin_centers.tolist(), "volumes": volumes.tolist()}

    # Point of Control (POC)
    poc_idx = int(np.argmax(volumes))
    poc_price = float(bin_centers[poc_idx])

    # Value Area (70% of total volume expanding symmetrically around POC)
    target_vol = total_volume * value_area_pct
    accum_vol = volumes[poc_idx]
    up_idx = poc_idx
    down_idx = poc_idx

    while accum_vol < target_vol and (up_idx < num_bins - 1 or down_idx > 0):
        up_vol = volumes[up_idx + 1] if up_idx < num_bins - 1 else -1.0
        down_vol = volumes[down_idx - 1] if down_idx > 0 else -1.0

        if up_vol >= down_vol and up_idx < num_bins - 1:
            up_idx += 1
            accum_vol += up_vol
        elif down_idx > 0:
            down_idx -= 1
            accum_vol += down_vol
        elif up_idx < num_bins - 1:
            up_idx += 1
            accum_vol += up_vol
        else:
            break

    vah_price = float(bin_centers[up_idx])
    val_price = float(bin_centers[down_idx])

    return {
        "poc": round(poc_price, 2),
        "vah": round(vah_price, 2),
        "val": round(val_price, 2),
        "bins": bin_centers.tolist(),
        "volumes": volumes.tolist(),
        "total_volume": float(total_volume),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Advanced Heikin-Ashi Smoothing
# ═══════════════════════════════════════════════════════════════════════

def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes smoothed Heikin-Ashi candlesticks to filter market chop
    and expose clean directional trend persistence.
    """
    if df.empty or len(df) < 2:
        return df.copy()

    ha_df = pd.DataFrame(index=df.index)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    ha_open = np.zeros(len(df), dtype=float)
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_high = np.maximum(df["high"].values, np.maximum(ha_open, ha_close.values))
    ha_low = np.minimum(df["low"].values, np.minimum(ha_open, ha_close.values))

    ha_df["timestamp"] = df["timestamp"]
    ha_df["open"] = ha_open
    ha_df["high"] = ha_high
    ha_df["low"] = ha_low
    ha_df["close"] = ha_close
    ha_df["volume"] = df["volume"]
    return ha_df


# ═══════════════════════════════════════════════════════════════════════
#  Anchored VWAP with Standard Deviation Envelopes
# ═══════════════════════════════════════════════════════════════════════

def compute_anchored_vwap_bands(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes Anchored VWAP along with +/-1 and +/-2 standard deviation bands.
    Used by veteran desks to identify institutional mean reversion levels.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    cum_pv = (typical_price * df["volume"]).cumsum()
    vwap = cum_pv / cum_vol

    # Variance calculation: sum(v * (p - vwap)^2) / sum(v)
    cum_dev_sq = (df["volume"] * ((typical_price - vwap) ** 2)).cumsum()
    vwap_std = np.sqrt(cum_dev_sq / cum_vol).fillna(0)

    upper_1 = vwap + 1.0 * vwap_std
    lower_1 = vwap - 1.0 * vwap_std
    upper_2 = vwap + 2.0 * vwap_std
    lower_2 = vwap - 2.0 * vwap_std

    return vwap, upper_1, lower_1, upper_2, lower_2
