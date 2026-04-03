"""
Execution Handler — Order routing.

Takes validated ``Signal`` objects, converts them into ``Order`` objects,
applies slippage / fee models, and submits.

Supports:
    - A simulated (paper) execution mode for backtesting / dev.
    - Placeholder for live broker integration (Angel One, Upstox, Zerodha, etc.)
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Optional

from config.logging_config import get_logger
from config.settings import RISK
from core.events import bus
from core.models import (
    Order,
    OrderStatus,
    OrderType,
    Side,
    Signal,
    SignalType,
)

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Slippage & Fee Model
# ═══════════════════════════════════════════════════════════════════════

def apply_slippage(price: float, side: Side) -> float:
    """
    Simulate slippage as configurable basis-points plus Gaussian noise.
    Buys get a *higher* fill; sells get *lower*.
    """
    bps = RISK.slippage_bps + random.gauss(0, 1)  # jitter ±1 bps
    factor = bps / 10_000
    if side == Side.BUY:
        return round(price * (1 + factor), 4)
    return round(price * (1 - factor), 4)


def compute_commission(quantity: float) -> float:
    return round(RISK.commission_per_share * quantity, 4)


# ═══════════════════════════════════════════════════════════════════════
#  Order Executor
# ═══════════════════════════════════════════════════════════════════════

class OrderExecutor:
    """
    Converts signals to broker orders.

    Parameters
    ----------
    live : bool
        When ``True``, orders are routed to Alpaca REST API.
        When ``False``, orders are filled immediately at the simulated price
        (used for backtesting & paper-dev).
    """

    def __init__(self, live: bool = False) -> None:
        self.live = live
        if live:
            log.warning(
                "Live broker integration not yet configured — "
                "orders will be simulated. Plug in Angel One / Upstox / Zerodha API here."
            )

    # ── Public API ────────────────────────────────────────────────────

    def execute(
        self, signal: Signal, quantity: float, current_price: float
    ) -> Optional[Order]:
        """
        Build and submit an order from a validated signal.

        Returns the filled ``Order`` object, or ``None`` if submission failed.
        """
        if quantity <= 0:
            log.warning("Invalid quantity %.2f for %s — skipping", quantity, signal.ticker)
            return None
        if current_price <= 0:
            log.warning("Invalid price %.4f for %s — skipping", current_price, signal.ticker)
            return None

        side = self._signal_to_side(signal.signal_type)
        if side is None:
            return None

        order = Order(
            ticker=signal.ticker,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            strategy_name=signal.strategy_name,
        )

        if self.live:
            # TODO: Plug in Angel One SmartAPI / Upstox / Zerodha Kite here
            log.info("LIVE mode — routing to simulated fill (broker not configured)")
        return self._simulate_fill(order, current_price)

    # ── Simulated fill ────────────────────────────────────────────────

    def _simulate_fill(self, order: Order, price: float) -> Order:
        fill_price = apply_slippage(price, order.side)
        commission = compute_commission(order.quantity)

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_qty = order.quantity
        order.commission = commission
        order.slippage = abs(fill_price - price) * order.quantity

        log.info(
            "SIM FILL | %s %s x %.0f @ %.2f (slp=%.2f, comm=%.2f)",
            order.side.value.upper(),
            order.ticker,
            order.quantity,
            fill_price,
            order.slippage,
            order.commission,
        )
        bus.publish("execution:fill", order)
        return order

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _signal_to_side(sig_type: SignalType) -> Optional[Side]:
        if sig_type in (SignalType.LONG_ENTRY,):
            return Side.BUY
        if sig_type in (SignalType.LONG_EXIT, SignalType.SHORT_ENTRY):
            return Side.SELL
        if sig_type == SignalType.SHORT_EXIT:
            return Side.BUY
        return None
