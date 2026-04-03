"""
Risk Management Core — "The Shield".

Responsibilities:
    1. Position sizing via Kelly Criterion (capped).
    2. Per-position stop-loss / take-profit computation.
    3. Daily drawdown circuit breaker.
    4. Pre-trade risk checks (max positions, max single exposure).
    5. Real-time PnL & drawdown tracking.

All public methods are thread-safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from config.logging_config import get_logger
from config.settings import RISK
from core.events import bus
from core.models import (
    Bar,
    Order,
    Position,
    PortfolioSnapshot,
    Side,
    Signal,
    SignalType,
)

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Risk Manager
# ═══════════════════════════════════════════════════════════════════════

class RiskManager:
    """Centralised risk gatekeeper."""

    def __init__(self, initial_equity: float = 100_000.0) -> None:
        self._lock = threading.Lock()

        # Account state
        self.equity = initial_equity
        self.cash = initial_equity
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity

        # Open positions keyed by ticker
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []

        # Circuit breaker
        self._halted = False
        self._halt_reason: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset_daily(self) -> None:
        """Call at the start of each trading day."""
        with self._lock:
            self.daily_start_equity = self.equity
            self._halted = False
            self._halt_reason = None
            log.info("Daily risk counters reset — equity %.2f", self.equity)

    # ── 1. Pre-trade risk checks ──────────────────────────────────────

    def approve_signal(self, signal: Signal, current_price: float) -> bool:
        """
        Return ``True`` if the signal passes all risk filters.
        Called *before* an order is created.
        """
        if current_price <= 0:
            log.warning("Invalid price %.4f for %s — rejecting", current_price, signal.ticker)
            return False

        with self._lock:
            if self._halted:
                log.warning("HALTED — rejecting signal %s", signal.id)
                return False

            # Check circuit breaker first
            self._check_circuit_breaker()
            if self._halted:
                return False

            ticker = signal.ticker

            # Block entries if at max positions
            if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
                if len(self.positions) >= RISK.max_open_positions:
                    log.warning("Max open positions — rejecting %s", ticker)
                    return False
                if ticker in self.positions:
                    log.warning("Already in position for %s", ticker)
                    return False

            return True

    # ── 2. Position sizing (Kelly Criterion) ──────────────────────────

    def compute_position_size(
        self,
        signal: Signal,
        current_price: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
    ) -> float:
        """
        Compute the number of shares/units to trade using the Kelly formula:

            f* = (p * b - q) / b

        where p = win_rate, q = 1-p, b = avg_win_loss_ratio.
        The result is capped at ``kelly_fraction_cap`` and further limited
        by ``max_position_pct`` of current equity.
        """
        p = min(max(win_rate, 0.01), 0.99)
        q = 1.0 - p
        b = max(avg_win_loss_ratio, 0.01)

        kelly_f = (p * b - q) / b
        kelly_f = max(kelly_f, 0.0)  # never negative
        kelly_f = min(kelly_f, RISK.kelly_fraction_cap)

        # Scale by signal strength
        kelly_f *= signal.strength

        max_dollar = self.equity * min(kelly_f, RISK.max_position_pct)
        shares = max_dollar / current_price if current_price > 0 else 0
        shares = int(shares)  # whole shares (adjust for crypto as needed)
        return float(max(shares, 0))

    # ── 3. Stop-loss / take-profit ────────────────────────────────────

    def compute_exit_levels(
        self,
        entry_price: float,
        side: Side,
        atr: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute stop-loss and take-profit prices.
        If *atr* (Average True Range) is provided, use ATR-based offsets;
        otherwise fall back to percentage-based defaults.
        """
        if atr and atr > 0:
            sl_offset = 2.0 * atr
            tp_offset = 3.0 * atr
        else:
            sl_offset = entry_price * RISK.default_stop_loss_pct
            tp_offset = entry_price * RISK.default_take_profit_pct

        if side == Side.BUY:
            return {
                "stop_loss": round(entry_price - sl_offset, 4),
                "take_profit": round(entry_price + tp_offset, 4),
            }
        else:
            return {
                "stop_loss": round(entry_price + sl_offset, 4),
                "take_profit": round(entry_price - tp_offset, 4),
            }

    # ── 4. Position & PnL tracking ────────────────────────────────────

    def open_position(self, order: Order) -> None:
        """Register a new filled entry order as an open position."""
        with self._lock:
            exits = self.compute_exit_levels(
                order.filled_price,
                order.side,
            )
            pos = Position(
                ticker=order.ticker,
                side=order.side,
                quantity=order.filled_qty,
                avg_entry_price=order.filled_price,
                current_price=order.filled_price,
                stop_loss=exits["stop_loss"],
                take_profit=exits["take_profit"],
                strategy_name=order.strategy_name,
            )
            self.positions[order.ticker] = pos
            self.cash -= order.filled_price * order.filled_qty
            log.info(
                "Opened %s %s × %.0f @ %.2f  SL=%.2f TP=%.2f",
                order.side.value, order.ticker, order.filled_qty,
                order.filled_price, pos.stop_loss, pos.take_profit,
            )

    def close_position(self, ticker: str, exit_price: float) -> Optional[float]:
        """Close an existing position and return realized PnL."""
        with self._lock:
            pos = self.positions.pop(ticker, None)
            if pos is None:
                return None
            direction = 1.0 if pos.side == Side.BUY else -1.0
            pnl = (exit_price - pos.avg_entry_price) * pos.quantity * direction
            self.cash += exit_price * pos.quantity
            self.equity += pnl

            self.closed_trades.append({
                "ticker": ticker,
                "side": pos.side.value,
                "entry": pos.avg_entry_price,
                "exit": exit_price,
                "qty": pos.quantity,
                "pnl": round(pnl, 2),
                "strategy": pos.strategy_name,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            })
            log.info("Closed %s -- PnL INR %.2f", ticker, pnl)
            return pnl

    def update_prices(self, bar: Bar) -> List[Signal]:
        """
        Update current price for a position and check SL / TP triggers.
        Returns exit signals if triggered.
        """
        exit_signals: List[Signal] = []
        with self._lock:
            pos = self.positions.get(bar.ticker)
            if pos is None:
                return exit_signals

            pos.current_price = bar.close
            pos.update_pnl()

            # Stop-loss check
            if pos.side == Side.BUY and bar.low <= (pos.stop_loss or 0):
                exit_signals.append(self._make_exit_signal(pos, "stop_loss"))
            elif pos.side == Side.SELL and bar.high >= (pos.stop_loss or float("inf")):
                exit_signals.append(self._make_exit_signal(pos, "stop_loss"))

            # Take-profit check
            if pos.side == Side.BUY and bar.high >= (pos.take_profit or float("inf")):
                exit_signals.append(self._make_exit_signal(pos, "take_profit"))
            elif pos.side == Side.SELL and bar.low <= (pos.take_profit or 0):
                exit_signals.append(self._make_exit_signal(pos, "take_profit"))

            # Update global equity
            self._recalc_equity()
            self._check_circuit_breaker()

        return exit_signals

    # ── 5. Circuit breaker ────────────────────────────────────────────

    def _check_circuit_breaker(self) -> None:
        """Halt trading if daily drawdown exceeds threshold."""
        if self.daily_start_equity <= 0:
            return
        daily_pnl_pct = (self.equity - self.daily_start_equity) / self.daily_start_equity
        if daily_pnl_pct <= -RISK.max_daily_drawdown_pct:
            self._halted = True
            self._halt_reason = (
                f"Daily drawdown {daily_pnl_pct:.2%} exceeds "
                f"limit {-RISK.max_daily_drawdown_pct:.2%}"
            )
            log.critical("🚨 CIRCUIT BREAKER TRIGGERED: %s", self._halt_reason)
            bus.publish("risk:circuit_breaker", self._halt_reason)

    # ── 6. Snapshot ───────────────────────────────────────────────────

    def snapshot(self) -> PortfolioSnapshot:
        with self._lock:
            self._recalc_equity()
            dd = 0.0
            if self.peak_equity > 0:
                dd = (self.peak_equity - self.equity) / self.peak_equity
            return PortfolioSnapshot(
                equity=round(self.equity, 2),
                cash=round(self.cash, 2),
                buying_power=round(self.cash, 2),
                daily_pnl=round(self.equity - self.daily_start_equity, 2),
                daily_pnl_pct=round(
                    (self.equity - self.daily_start_equity)
                    / self.daily_start_equity
                    * 100,
                    3,
                )
                if self.daily_start_equity
                else 0.0,
                open_positions=len(self.positions),
                peak_equity=round(self.peak_equity, 2),
                drawdown_pct=round(dd * 100, 3),
            )

    # ── helpers ───────────────────────────────────────────────────────

    def _recalc_equity(self) -> None:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        invested = sum(
            p.avg_entry_price * p.quantity for p in self.positions.values()
        )
        self.equity = self.cash + invested + unrealized
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    @staticmethod
    def _make_exit_signal(pos: Position, reason: str) -> Signal:
        sig_type = (
            SignalType.LONG_EXIT if pos.side == Side.BUY else SignalType.SHORT_EXIT
        )
        return Signal(
            ticker=pos.ticker,
            signal_type=sig_type,
            strategy_name=f"risk:{reason}",
            strength=1.0,
            metadata={"reason": reason, "trigger_price": pos.current_price},
        )
