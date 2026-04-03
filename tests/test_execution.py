"""Tests for the Order Executor — simulated fills, slippage, validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import Signal, SignalType, Side, OrderStatus
from execution.order_executor import (
    OrderExecutor,
    apply_slippage,
    compute_commission,
)


def _make_signal(ticker="AAPL", sig_type=SignalType.LONG_ENTRY):
    return Signal(
        ticker=ticker,
        signal_type=sig_type,
        strategy_name="test",
        strength=0.8,
    )


# ────────────────────────────────────────────────────────────────────
#  Slippage
# ────────────────────────────────────────────────────────────────────

class TestSlippage:
    def test_buy_slippage_positive(self):
        """Buys should fill at or above the reference price."""
        prices = [apply_slippage(100.0, Side.BUY) for _ in range(100)]
        assert sum(p >= 100.0 for p in prices) > 80  # most should be above

    def test_sell_slippage_negative(self):
        """Sells should fill at or below the reference price."""
        prices = [apply_slippage(100.0, Side.SELL) for _ in range(100)]
        assert sum(p <= 100.0 for p in prices) > 80


class TestCommission:
    def test_zero_commission(self):
        assert compute_commission(100) == 0.0


# ────────────────────────────────────────────────────────────────────
#  Executor
# ────────────────────────────────────────────────────────────────────

class TestOrderExecutor:
    def test_simulated_fill(self):
        exe = OrderExecutor(live=False)
        sig = _make_signal()
        order = exe.execute(sig, quantity=10, current_price=150.0)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 10
        assert order.filled_price > 0

    def test_reject_zero_quantity(self):
        exe = OrderExecutor(live=False)
        sig = _make_signal()
        order = exe.execute(sig, quantity=0, current_price=150.0)
        assert order is None

    def test_reject_negative_price(self):
        exe = OrderExecutor(live=False)
        sig = _make_signal()
        order = exe.execute(sig, quantity=10, current_price=-1.0)
        assert order is None

    def test_neutral_signal_returns_none(self):
        exe = OrderExecutor(live=False)
        sig = _make_signal(sig_type=SignalType.NEUTRAL)
        order = exe.execute(sig, quantity=10, current_price=150.0)
        assert order is None

    def test_sell_signal_generates_sell_order(self):
        exe = OrderExecutor(live=False)
        sig = _make_signal(sig_type=SignalType.LONG_EXIT)
        order = exe.execute(sig, quantity=5, current_price=150.0)
        assert order is not None
        assert order.side == Side.SELL
