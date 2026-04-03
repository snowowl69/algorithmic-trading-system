"""
Sentiment-Driven Strategy.

Trades *exclusively* on extreme NLP sentiment scores.  When the aggregate
sentiment for a ticker crosses the ``sentiment_extreme_score`` threshold
in either direction, an entry signal is generated.
"""

from __future__ import annotations

from typing import Dict, Optional

from config.settings import STRATEGY
from core.models import Bar, Sentiment, SentimentResult, Signal, SignalType
from strategies.base_strategy import BaseStrategy


class SentimentStrategy(BaseStrategy):
    name = "sentiment_extreme"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = STRATEGY.sentiment_extreme_score
        # Track whether we already signalled for this ticker/direction
        self._active_signals: Dict[str, SignalType] = {}
        # Aggregate EMA of scores per ticker
        self._agg_scores: Dict[str, float] = {}
        self._ema_alpha = 0.4

    # ── Sentiment is the only trigger ─────────────────────────────────
    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._record_bar(bar)
        # Check if we should exit an active position by decay
        ticker = bar.ticker
        agg = self._agg_scores.get(ticker, 0.0)
        active = self._active_signals.get(ticker)

        if active == SignalType.LONG_ENTRY and agg < 0.1:
            del self._active_signals[ticker]
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_EXIT,
                strategy_name=self.name,
                strength=0.4,
                metadata={"reason": "sentiment_decay", "agg_score": agg},
            )
        if active == SignalType.SHORT_ENTRY and agg > -0.1:
            del self._active_signals[ticker]
            return Signal(
                ticker=ticker,
                signal_type=SignalType.SHORT_EXIT,
                strategy_name=self.name,
                strength=0.4,
                metadata={"reason": "sentiment_decay", "agg_score": agg},
            )
        return None

    def on_sentiment(self, result: SentimentResult) -> Optional[Signal]:
        ticker = result.ticker
        score = result.score

        # Update exponential moving average
        prev = self._agg_scores.get(ticker, 0.0)
        agg = self._ema_alpha * score + (1 - self._ema_alpha) * prev
        self._agg_scores[ticker] = agg

        active = self._active_signals.get(ticker)

        # ── Extreme bullish ───────────────────────────────────────────
        if agg >= self.threshold and active != SignalType.LONG_ENTRY:
            self._active_signals[ticker] = SignalType.LONG_ENTRY
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_ENTRY,
                strategy_name=self.name,
                strength=min(abs(agg), 1.0),
                metadata={
                    "agg_score": round(agg, 4),
                    "headline": result.headline[:100],
                    "sentiment": result.sentiment.value,
                },
            )

        # ── Extreme bearish ───────────────────────────────────────────
        if agg <= -self.threshold and active != SignalType.SHORT_ENTRY:
            self._active_signals[ticker] = SignalType.SHORT_ENTRY
            return Signal(
                ticker=ticker,
                signal_type=SignalType.SHORT_ENTRY,
                strategy_name=self.name,
                strength=min(abs(agg), 1.0),
                metadata={
                    "agg_score": round(agg, 4),
                    "headline": result.headline[:100],
                    "sentiment": result.sentiment.value,
                },
            )

        return None
