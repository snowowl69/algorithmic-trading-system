"""
Shared domain models used across every module.
All models are immutable dataclasses for thread safety.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalType(str, Enum):
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"
    NEUTRAL = "neutral"


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ── Data Models ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Bar:
    """Single OHLCV bar / candle."""
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


@dataclass(frozen=True)
class Signal:
    """A trading signal emitted by a strategy."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str = ""
    signal_type: SignalType = SignalType.NEUTRAL
    strategy_name: str = ""
    strength: float = 0.0          # 0.0 – 1.0 confidence
    metadata: dict = field(default_factory=dict)


@dataclass
class Order:
    """Broker order representation."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: str = ""
    side: Side = Side.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy_name: str = ""


@dataclass
class Position:
    """Current open position."""
    ticker: str = ""
    side: Side = Side.BUY
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""

    def update_pnl(self) -> None:
        diff = self.current_price - self.avg_entry_price
        direction = 1.0 if self.side == Side.BUY else -1.0
        self.unrealized_pnl = diff * self.quantity * direction


@dataclass(frozen=True)
class SentimentResult:
    """Output of the NLP sentiment pipeline."""
    ticker: str
    headline: str
    source: str
    published_at: datetime
    sentiment: Sentiment
    score: float  # –1.0 (very bearish) … +1.0 (very bullish)


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of account state."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    open_positions: int = 0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
