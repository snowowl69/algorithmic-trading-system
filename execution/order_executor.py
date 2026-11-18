"""
Execution Handler — Order routing.

Takes validated ``Signal`` objects, converts them into ``Order`` objects,
applies slippage / fee models, and submits.

Supports:
    - A simulated (paper) execution mode for backtesting / dev.
    - Placeholder for live broker integration (Angel One, Upstox, Zerodha, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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

def apply_slippage(
    price: float, side: Side, quantity: float = 0.0, avg_daily_volume: float = 0.0
) -> float:
    """
    Simulate slippage as configurable basis-points plus Gaussian noise,
    **scaled by market impact** when volume data is available.

    Market impact: if you're trading 1% of ADV, slippage is amplified.
    This prevents the backtest from assuming infinite liquidity.

    Buys get a *higher* fill; sells get *lower*.
    """
    bps = RISK.slippage_bps + random.gauss(0, 1)  # jitter ±1 bps

    # Volume-aware market impact multiplier
    if avg_daily_volume > 0 and quantity > 0:
        participation_rate = quantity / avg_daily_volume
        impact_multiplier = 1.0 + RISK.volume_impact_factor * participation_rate
        bps *= impact_multiplier

    factor = bps / 10_000
    if side == Side.BUY:
        return round(price * (1 + factor), 4)
    return round(price * (1 - factor), 4)


def compute_commission(quantity: float) -> float:
    return round(RISK.commission_per_share * quantity, 4)


# ═══════════════════════════════════════════════════════════════════════
#  Order Executor
# ═══════════════════════════════════════════════════════════════════════

class BaseBrokerAdapter(ABC):
    """Abstract base class for all broker execution adapters."""
    @abstractmethod
    def place_order(self, order: Order, price: float) -> Optional[Order]:
        pass

class ZerodhaKiteAdapter(BaseBrokerAdapter):
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            log.info("Zerodha Kite Connect initialized")
        except ImportError:
            log.error("kiteconnect package not installed. Run: pip install kiteconnect")
            self.kite = None

    def place_order(self, order: Order, price: float) -> Optional[Order]:
        if not self.kite:
            log.error("Zerodha Kite adapter not active.")
            return None
        # Convert internal order to Kite format
        kite_side = "BUY" if order.side == Side.BUY else "SELL"
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=order.ticker,
                transaction_type=kite_side,
                quantity=int(order.quantity),
                product=self.kite.PRODUCT_MIS, # Intraday
                order_type=self.kite.ORDER_TYPE_MARKET
            )
            log.info("Zerodha Live Order placed: %s", order_id)
            order.status = OrderStatus.FILLED
            order.filled_price = price
            order.filled_qty = order.quantity
            return order
        except Exception as e:
            log.error("Zerodha execution failed: %s", e)
            return None

class UpstoxAdapter(BaseBrokerAdapter):
    def place_order(self, order: Order, price: float) -> Optional[Order]:
        log.info("Upstox live routing invoked for %s", order.ticker)
        # Stub for Upstox v2 API
        return None

class OrderExecutor:
    """
    Converts signals to broker orders.

    Parameters
    ----------
    live : bool
        When ``True``, orders are routed to the configured BrokerAdapter.
        When ``False``, orders are filled immediately at the simulated price
        (used for backtesting & paper-dev).
    """

    def __init__(self, live: bool = False, broker_adapter: Optional[BaseBrokerAdapter] = None) -> None:
        self.live = live
        self.broker_adapter = broker_adapter
        self.total_slippage_cost: float = 0.0
        self.total_commission_cost: float = 0.0
        self.total_orders_executed: int = 0
        if live:
            if not broker_adapter:
                log.warning(
                    "Live mode enabled but no BrokerAdapter provided. "
                    "Orders will be SIMULATED. Please configure ZerodhaKiteAdapter or UpstoxAdapter."
                )
            else:
                log.info("LIVE EXECUTOR ACTIVE using %s", broker_adapter.__class__.__name__)

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

        if self.live and self.broker_adapter:
            return self.broker_adapter.place_order(order, current_price)
        
        return self._simulate_fill(order, current_price)

    # ── Simulated fill ────────────────────────────────────────────────

    def _simulate_fill(self, order: Order, price: float) -> Order:
        fill_price = apply_slippage(price, order.side, order.quantity)
        commission = compute_commission(order.quantity)

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_qty = order.quantity
        order.commission = commission
        order.slippage = abs(fill_price - price) * order.quantity

        # Track cumulative execution costs
        self.total_slippage_cost += order.slippage
        self.total_commission_cost += commission
        self.total_orders_executed += 1

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
