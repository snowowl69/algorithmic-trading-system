"""
Mean-Reversion Strategy — Z-Score based.

Uses a rolling lookback window to compute a z-score of the current price
relative to its recent mean. Enters when price deviates beyond a threshold
and exits when it reverts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config.settings import STRATEGY
from core.models import Bar, SentimentResult, Signal, SignalType
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion_zscore"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = STRATEGY.mean_rev_lookback
        self.z_entry = STRATEGY.mean_rev_z_entry
        self.z_exit = STRATEGY.mean_rev_z_exit
        self._in_position: dict[str, SignalType] = {}

    # ── core logic ────────────────────────────────────────────────────
    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._record_bar(bar)
        ticker = bar.ticker

        if not self._has_enough_bars(ticker, self.lookback):
            return None

        closes = self._closes(ticker)[-self.lookback:]
        mean = float(np.mean(closes))
        std = float(np.std(closes, ddof=1))
        if std == 0:
            return None

        z = (bar.close - mean) / std
        current = self._in_position.get(ticker)

        # ── Entry signals ─────────────────────────────────────────────
        if current is None:
            if z < -self.z_entry:
                # Price abnormally low → mean-revert long
                self._in_position[ticker] = SignalType.LONG_ENTRY
                return Signal(
                    ticker=ticker,
                    signal_type=SignalType.LONG_ENTRY,
                    strategy_name=self.name,
                    strength=min(abs(z) / 3.0, 1.0),
                    metadata={"z_score": round(z, 3), "mean": mean, "std": std},
                )
            if z > self.z_entry:
                # Price abnormally high → mean-revert short
                self._in_position[ticker] = SignalType.SHORT_ENTRY
                return Signal(
                    ticker=ticker,
                    signal_type=SignalType.SHORT_ENTRY,
                    strategy_name=self.name,
                    strength=min(abs(z) / 3.0, 1.0),
                    metadata={"z_score": round(z, 3), "mean": mean, "std": std},
                )

        # ── Exit signals ──────────────────────────────────────────────
        if current == SignalType.LONG_ENTRY and abs(z) < self.z_exit:
            del self._in_position[ticker]
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_EXIT,
                strategy_name=self.name,
                strength=0.5,
                metadata={"z_score": round(z, 3)},
            )

        if current == SignalType.SHORT_ENTRY and abs(z) < self.z_exit:
            del self._in_position[ticker]
            return Signal(
                ticker=ticker,
                signal_type=SignalType.SHORT_EXIT,
                strategy_name=self.name,
                strength=0.5,
                metadata={"z_score": round(z, 3)},
            )

        return None

    def on_sentiment(self, result: SentimentResult) -> Optional[Signal]:
        # Mean reversion does not use sentiment
        return None
