"""
Abstract base class that every strategy must implement.

The framework calls ``on_bar()`` for each new price bar and ``on_sentiment()``
for each new NLP result. Strategies return ``Signal`` objects (or ``None``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import numpy as np

from core.models import Bar, SentimentResult, Signal


class BaseStrategy(ABC):
    """
    All strategies inherit from this class.

    Subclasses must implement:
        - ``on_bar(bar)``     → Optional[Signal]
        - ``on_sentiment(sr)`` → Optional[Signal]
    """

    name: str = "base"

    def __init__(self, tickers: Optional[List[str]] = None) -> None:
        self.tickers = tickers or []
        # rolling bar buffer per ticker
        self._bar_history: Dict[str, Deque[Bar]] = defaultdict(
            lambda: deque(maxlen=200)
        )

    # ── abstract hooks ────────────────────────────────────────────────
    @abstractmethod
    def on_bar(self, bar: Bar) -> Optional[Signal]:
        """Process a new price bar and optionally return a trading signal."""
        ...

    @abstractmethod
    def on_sentiment(self, result: SentimentResult) -> Optional[Signal]:
        """Process a new sentiment result and optionally return a signal."""
        ...

    # ── helpers for subclasses ────────────────────────────────────────
    def _record_bar(self, bar: Bar) -> None:
        self._bar_history[bar.ticker].append(bar)

    def _closes(self, ticker: str) -> np.ndarray:
        """Return a numpy array of recent close prices for *ticker*."""
        return np.array([b.close for b in self._bar_history[ticker]])

    def _has_enough_bars(self, ticker: str, n: int) -> bool:
        return len(self._bar_history[ticker]) >= n
