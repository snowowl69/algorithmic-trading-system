"""
Momentum / Trend-Following Strategy — MA Crossover.

Generates LONG_ENTRY when the fast EMA crosses above the slow EMA,
and LONG_EXIT when it crosses below.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config.settings import STRATEGY
from core.models import Bar, SentimentResult, Signal, SignalType
from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    name = "momentum_ma_crossover"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fast = STRATEGY.momentum_fast_window
        self.slow = STRATEGY.momentum_slow_window
        self._prev_fast: dict[str, float] = {}
        self._prev_slow: dict[str, float] = {}

    # ── core logic ────────────────────────────────────────────────────
    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._record_bar(bar)
        ticker = bar.ticker

        if not self._has_enough_bars(ticker, self.slow + 1):
            return None

        closes = self._closes(ticker)
        fast_ema = self._ema(closes, self.fast)
        slow_ema = self._ema(closes, self.slow)

        prev_fast = self._prev_fast.get(ticker)
        prev_slow = self._prev_slow.get(ticker)

        self._prev_fast[ticker] = fast_ema
        self._prev_slow[ticker] = slow_ema

        if prev_fast is None or prev_slow is None:
            return None

        # Bullish crossover
        if prev_fast <= prev_slow and fast_ema > slow_ema:
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_ENTRY,
                strategy_name=self.name,
                strength=min((fast_ema - slow_ema) / slow_ema * 100, 1.0),
                metadata={"fast_ema": fast_ema, "slow_ema": slow_ema},
            )
        # Bearish crossover
        if prev_fast >= prev_slow and fast_ema < slow_ema:
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_EXIT,
                strategy_name=self.name,
                strength=min((slow_ema - fast_ema) / slow_ema * 100, 1.0),
                metadata={"fast_ema": fast_ema, "slow_ema": slow_ema},
            )

        return None

    def on_sentiment(self, result: SentimentResult) -> Optional[Signal]:
        # Momentum strategy does not react to sentiment
        return None

    # ── utilities ─────────────────────────────────────────────────────
    @staticmethod
    def _ema(data: np.ndarray, window: int) -> float:
        """Compute the latest Exponential Moving Average value using the standard recursive formula."""
        alpha = 2.0 / (window + 1)
        ema_val = float(data[0])
        for price in data[1:]:
            ema_val = alpha * float(price) + (1.0 - alpha) * ema_val
        return ema_val
