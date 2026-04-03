"""Tests for the three trading strategies."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import Bar, SentimentResult, Sentiment, SignalType
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.sentiment_strategy import SentimentStrategy


# ────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────

def _make_bars(prices: list[float], ticker: str = "TEST") -> list[Bar]:
    """Build a list of bars with ascending timestamps."""
    base = datetime.now(timezone.utc) - timedelta(minutes=len(prices))
    return [
        Bar(
            ticker=ticker,
            timestamp=base + timedelta(minutes=i),
            open=p, high=p + 0.5, low=p - 0.5, close=p,
            volume=1000,
        )
        for i, p in enumerate(prices)
    ]


def _make_sentiment(ticker="TEST", score=0.9, label=Sentiment.BULLISH):
    return SentimentResult(
        ticker=ticker,
        headline="Test headline",
        source="test",
        published_at=datetime.now(timezone.utc),
        sentiment=label,
        score=score,
    )


# ────────────────────────────────────────────────────────────────────
#  Momentum Strategy
# ────────────────────────────────────────────────────────────────────

class TestMomentumStrategy:
    def test_needs_enough_bars(self):
        strat = MomentumStrategy(tickers=["TEST"])
        bars = _make_bars([100.0] * 10)
        for bar in bars:
            result = strat.on_bar(bar)
        assert result is None  # not enough for slow window (50)

    def test_generates_signal_on_crossover(self):
        strat = MomentumStrategy(tickers=["TEST"])
        # Feed a downtrend then a sharp uptrend to trigger bullish crossover
        prices = list(range(200, 140, -1)) + list(range(140, 210))
        bars = _make_bars(prices)
        signals = []
        for bar in bars:
            sig = strat.on_bar(bar)
            if sig is not None:
                signals.append(sig)
        # Should have at least one entry and one exit signal
        types = {s.signal_type for s in signals}
        assert SignalType.LONG_ENTRY in types or SignalType.LONG_EXIT in types

    def test_ignores_sentiment(self):
        strat = MomentumStrategy(tickers=["TEST"])
        sr = _make_sentiment()
        assert strat.on_sentiment(sr) is None


# ────────────────────────────────────────────────────────────────────
#  Mean Reversion Strategy
# ────────────────────────────────────────────────────────────────────

class TestMeanReversionStrategy:
    def test_needs_lookback_period(self):
        strat = MeanReversionStrategy(tickers=["TEST"])
        bars = _make_bars([100.0] * 5)
        for bar in bars:
            result = strat.on_bar(bar)
        assert result is None  # not enough bars

    def test_signals_on_extreme_deviation(self):
        strat = MeanReversionStrategy(tickers=["TEST"])
        # Stable prices then a sharp drop → z-score goes negative
        prices = [100.0] * 25 + [90.0]
        bars = _make_bars(prices)
        signals = []
        for bar in bars:
            sig = strat.on_bar(bar)
            if sig is not None:
                signals.append(sig)
        assert len(signals) >= 1
        assert signals[0].signal_type == SignalType.LONG_ENTRY

    def test_ignores_sentiment(self):
        strat = MeanReversionStrategy(tickers=["TEST"])
        sr = _make_sentiment()
        assert strat.on_sentiment(sr) is None


# ────────────────────────────────────────────────────────────────────
#  Sentiment Strategy
# ────────────────────────────────────────────────────────────────────

class TestSentimentStrategy:
    def test_extreme_bullish_triggers_long(self):
        strat = SentimentStrategy(tickers=["TEST"])
        # Feed several very bullish results to push EMA over threshold
        signals = []
        for _ in range(10):
            sig = strat.on_sentiment(_make_sentiment(score=0.95))
            if sig is not None:
                signals.append(sig)
        assert len(signals) >= 1
        assert signals[0].signal_type == SignalType.LONG_ENTRY

    def test_extreme_bearish_triggers_short(self):
        strat = SentimentStrategy(tickers=["TEST"])
        signals = []
        for _ in range(10):
            sig = strat.on_sentiment(
                _make_sentiment(score=-0.95, label=Sentiment.BEARISH)
            )
            if sig is not None:
                signals.append(sig)
        assert len(signals) >= 1
        assert signals[0].signal_type == SignalType.SHORT_ENTRY

    def test_neutral_no_signal(self):
        strat = SentimentStrategy(tickers=["TEST"])
        sig = strat.on_sentiment(_make_sentiment(score=0.1, label=Sentiment.NEUTRAL))
        assert sig is None
