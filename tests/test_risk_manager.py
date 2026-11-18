"""Tests for the Risk Management Core — circuit breaker, Kelly sizing, SL/TP."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import Bar, Order, OrderStatus, Side, Signal, SignalType
from risk.risk_manager import RiskManager
from config.settings import RISK


# ────────────────────────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def rm():
    """Fresh RiskManager with 100k equity."""
    mgr = RiskManager(initial_equity=100_000.0)
    mgr.reset_daily()
    return mgr


def _make_bar(ticker="AAPL", close=150.0, low=149.0, high=151.0):
    return Bar(
        ticker=ticker,
        timestamp=datetime.now(timezone.utc),
        open=150.0, high=high, low=low, close=close,
        volume=10_000,
    )


def _make_signal(ticker="AAPL", sig_type=SignalType.LONG_ENTRY, strength=0.8):
    return Signal(
        ticker=ticker,
        signal_type=sig_type,
        strategy_name="test_strat",
        strength=strength,
    )


def _make_filled_order(ticker="AAPL", side=Side.BUY, price=150.0, qty=10.0):
    return Order(
        ticker=ticker,
        side=side,
        quantity=qty,
        filled_price=price,
        filled_qty=qty,
        status=OrderStatus.FILLED,
        strategy_name="test_strat",
    )


# ────────────────────────────────────────────────────────────────────
#  Position sizing
# ────────────────────────────────────────────────────────────────────

class TestKellySizing:
    def test_positive_size(self, rm):
        sig = _make_signal(strength=0.8)
        size = rm.compute_position_size(sig, current_price=150.0)
        assert size > 0

    def test_zero_strength_gives_zero(self, rm):
        sig = _make_signal(strength=0.0)
        size = rm.compute_position_size(sig, current_price=150.0)
        assert size == 0

    def test_zero_price_gives_zero(self, rm):
        sig = _make_signal(strength=0.8)
        size = rm.compute_position_size(sig, current_price=0.0)
        assert size == 0

    def test_size_capped_by_max_position(self, rm):
        sig = _make_signal(strength=1.0)
        size = rm.compute_position_size(
            sig, current_price=10.0, win_rate=0.99, avg_win_loss_ratio=10.0
        )
        max_shares = (rm.equity * RISK.max_position_pct) / 10.0
        assert size <= max_shares + 1  # +1 for int rounding


# ────────────────────────────────────────────────────────────────────
#  Signal approval
# ────────────────────────────────────────────────────────────────────

class TestApproveSignal:
    def test_basic_approval(self, rm):
        sig = _make_signal()
        assert rm.approve_signal(sig, current_price=150.0) is True

    def test_reject_on_invalid_price(self, rm):
        sig = _make_signal()
        assert rm.approve_signal(sig, current_price=0.0) is False
        assert rm.approve_signal(sig, current_price=-10.0) is False

    def test_reject_duplicate_ticker(self, rm):
        order = _make_filled_order()
        rm.open_position(order)
        sig = _make_signal(ticker="AAPL")
        assert rm.approve_signal(sig, current_price=150.0) is False

    def test_reject_when_halted(self, rm):
        rm._halted = True
        sig = _make_signal()
        assert rm.approve_signal(sig, current_price=150.0) is False

    def test_reject_max_positions(self, rm):
        for i in range(RISK.max_open_positions):
            order = _make_filled_order(ticker=f"TK{i}", price=100.0, qty=1)
            rm.open_position(order)
        sig = _make_signal(ticker="NEW")
        assert rm.approve_signal(sig, current_price=100.0) is False


# ────────────────────────────────────────────────────────────────────
#  Stop-loss / take-profit
# ────────────────────────────────────────────────────────────────────

class TestExitLevels:
    def test_long_exit_levels(self, rm):
        levels = rm.compute_exit_levels(100.0, Side.BUY)
        assert levels["stop_loss"] < 100.0
        assert levels["take_profit"] > 100.0

    def test_short_exit_levels(self, rm):
        levels = rm.compute_exit_levels(100.0, Side.SELL)
        assert levels["stop_loss"] > 100.0
        assert levels["take_profit"] < 100.0

    def test_atr_based_levels(self, rm):
        levels = rm.compute_exit_levels(100.0, Side.BUY, atr=2.0)
        assert levels["stop_loss"] == 96.0    # 100 - 2*2
        assert levels["take_profit"] == 106.0  # 100 + 3*2


# ────────────────────────────────────────────────────────────────────
#  Stop-loss trigger
# ────────────────────────────────────────────────────────────────────

class TestStopLossTrigger:
    def test_long_stop_loss_fires(self, rm):
        order = _make_filled_order(price=100.0, qty=10)
        rm.open_position(order)
        sl = rm.positions["AAPL"].stop_loss
        # Bar dips below stop-loss
        bar = _make_bar(close=sl - 1, low=sl - 2, high=sl + 1)
        exits = rm.update_prices(bar)
        assert len(exits) >= 1
        assert exits[0].signal_type == SignalType.LONG_EXIT

    def test_long_take_profit_fires(self, rm):
        order = _make_filled_order(price=100.0, qty=10)
        rm.open_position(order)
        tp = rm.positions["AAPL"].take_profit
        bar = _make_bar(close=tp + 1, low=tp - 1, high=tp + 2)
        exits = rm.update_prices(bar)
        assert any(e.metadata.get("reason") == "take_profit" for e in exits)


# ────────────────────────────────────────────────────────────────────
#  Circuit breaker
# ────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_triggers_on_drawdown(self, rm):
        # Simulate 5% loss
        rm.equity = rm.daily_start_equity * 0.95
        rm._check_circuit_breaker()
        assert rm.is_halted is True

    def test_no_trigger_within_threshold(self, rm):
        rm.equity = rm.daily_start_equity * 0.98  # 2% loss < 3% limit
        rm._check_circuit_breaker()
        assert rm.is_halted is False

    def test_reset_clears_halt(self, rm):
        rm._halted = True
        rm.reset_daily()
        assert rm.is_halted is False


# ────────────────────────────────────────────────────────────────────
#  Position open / close / PnL
# ────────────────────────────────────────────────────────────────────

class TestPositionTracking:
    def test_open_and_close_long(self, rm):
        order = _make_filled_order(price=100.0, qty=10)
        rm.open_position(order)
        assert "AAPL" in rm.positions

        pnl = rm.close_position("AAPL", exit_price=110.0)
        assert pnl == pytest.approx(100.0, abs=1.0)  # (110-100)*10
        assert "AAPL" not in rm.positions

    def test_open_and_close_short(self, rm):
        order = _make_filled_order(side=Side.SELL, price=200.0, qty=5)
        rm.open_position(order)

        pnl = rm.close_position("AAPL", exit_price=190.0)
        assert pnl == pytest.approx(50.0, abs=1.0)  # (200-190)*5

    def test_close_nonexistent_returns_none(self, rm):
        assert rm.close_position("ZZZ", 100.0) is None

    def test_snapshot_fields(self, rm):
        snap = rm.snapshot()
        assert snap.equity == pytest.approx(100_000.0, abs=0.01)
        assert snap.open_positions == 0
