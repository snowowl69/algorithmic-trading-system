"""
VWAP Cross Strategy.

Generates LONG_ENTRY when price crosses above VWAP with above-average
volume, and LONG_EXIT when price crosses below VWAP with above-average
volume.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from core.models import Bar, SentimentResult, Signal, SignalType
from strategies.base_strategy import BaseStrategy


class VWAPStrategy(BaseStrategy):
    """Volume-Weighted Average Price crossover strategy.

    Tracks cumulative VWAP per ticker across bars and fires signals
    when price crosses above/below VWAP with confirming volume.

    Parameters
    ----------
    volume_factor : float
        Minimum multiple of average volume required to confirm a signal
        (default 1.5).
    min_bars : int
        Minimum number of bars before the strategy starts producing
        signals (default 10).
    """

    name = "vwap_cross"

    def __init__(
        self,
        volume_factor: float = 1.5,
        min_bars: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.volume_factor = volume_factor
        self.min_bars = min_bars

        # Cumulative VWAP state per ticker
        self._cum_volume: Dict[str, float] = defaultdict(float)
        self._cum_vp: Dict[str, float] = defaultdict(float)       # sum(price * volume)
        self._prev_close: Dict[str, float] = {}
        self._prev_vwap: Dict[str, float] = {}
        self._volume_history: Dict[str, List[float]] = defaultdict(list)

    # ---- core logic ---------------------------------------------------

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        """Process a new price bar and return a signal on VWAP crossover."""
        self._record_bar(bar)
        ticker = bar.ticker

        # Accumulate VWAP components
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self._cum_volume[ticker] += bar.volume
        self._cum_vp[ticker] += typical_price * bar.volume
        self._volume_history[ticker].append(bar.volume)

        if self._cum_volume[ticker] == 0:
            return None

        vwap = self._cum_vp[ticker] / self._cum_volume[ticker]

        if not self._has_enough_bars(ticker, self.min_bars):
            self._prev_close[ticker] = bar.close
            self._prev_vwap[ticker] = vwap
            return None

        prev_close = self._prev_close.get(ticker)
        prev_vwap = self._prev_vwap.get(ticker)

        self._prev_close[ticker] = bar.close
        self._prev_vwap[ticker] = vwap

        if prev_close is None or prev_vwap is None:
            return None

        # Volume confirmation
        avg_volume = float(np.mean(self._volume_history[ticker]))
        volume_ok = bar.volume > self.volume_factor * avg_volume

        if not volume_ok:
            return None

        # Strength scales between 0.6 and 0.9 based on volume spike
        volume_ratio = bar.volume / max(avg_volume, 1.0)
        strength = min(0.6 + (volume_ratio - self.volume_factor) * 0.1, 0.9)
        strength = max(strength, 0.6)

        # Bullish crossover: price crosses above VWAP
        if prev_close <= prev_vwap and bar.close > vwap:
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_ENTRY,
                strategy_name=self.name,
                strength=strength,
                metadata={
                    "vwap": round(vwap, 4),
                    "close": bar.close,
                    "volume_ratio": round(volume_ratio, 2),
                },
            )

        # Bearish crossover: price crosses below VWAP
        if prev_close >= prev_vwap and bar.close < vwap:
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_EXIT,
                strategy_name=self.name,
                strength=strength,
                metadata={
                    "vwap": round(vwap, 4),
                    "close": bar.close,
                    "volume_ratio": round(volume_ratio, 2),
                },
            )

        return None

    def on_sentiment(self, result: SentimentResult) -> Optional[Signal]:
        """VWAP strategy does not react to sentiment."""
        return None
