"""Tests for core data models."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import (
    Bar, Signal, Order, Position, Side, SignalType,
    OrderType, OrderStatus, Sentiment, SentimentResult,
)


class TestBar:
    def test_immutable(self):
        bar = Bar(
            ticker="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=100, high=105, low=99, close=102, volume=1000,
        )
        with pytest.raises(AttributeError):
            bar.close = 200  # frozen

    def test_optional_vwap(self):
        bar = Bar(
            ticker="X",
            timestamp=datetime.now(timezone.utc),
            open=1, high=2, low=0.5, close=1.5, volume=10,
        )
        assert bar.vwap is None


class TestSignal:
    def test_defaults(self):
        sig = Signal()
        assert sig.signal_type == SignalType.NEUTRAL
        assert sig.strength == 0.0
        assert len(sig.id) == 12

    def test_immutable(self):
        sig = Signal(ticker="AAPL")
        with pytest.raises(AttributeError):
            sig.ticker = "MSFT"


class TestPosition:
    def test_pnl_long(self):
        pos = Position(
            ticker="AAPL", side=Side.BUY,
            quantity=10, avg_entry_price=100, current_price=110,
        )
        pos.update_pnl()
        assert pos.unrealized_pnl == pytest.approx(100.0)

    def test_pnl_short(self):
        pos = Position(
            ticker="AAPL", side=Side.SELL,
            quantity=10, avg_entry_price=100, current_price=90,
        )
        pos.update_pnl()
        assert pos.unrealized_pnl == pytest.approx(100.0)

    def test_pnl_negative(self):
        pos = Position(
            ticker="X", side=Side.BUY,
            quantity=5, avg_entry_price=200, current_price=190,
        )
        pos.update_pnl()
        assert pos.unrealized_pnl == pytest.approx(-50.0)


class TestEnums:
    def test_side_values(self):
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"

    def test_sentiment_values(self):
        assert Sentiment.BULLISH.value == "bullish"
        assert Sentiment.BEARISH.value == "bearish"
        assert Sentiment.NEUTRAL.value == "neutral"
