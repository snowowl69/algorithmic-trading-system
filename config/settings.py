"""
Central configuration for the Algo Trading system — Indian Markets (NSE / BSE / MCX).
All secrets are loaded from environment variables; defaults are for development only.
Everything is in INR (₹).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env file if present (before reading env vars)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars only


# ── Database path ─────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "trades.db"


@dataclass(frozen=True)
class NLPConfig:
    """NLP / FinBERT pipeline settings."""
    model_name: str = "ProsusAI/finbert"
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    news_api_url: str = "https://newsapi.org/v2/everything"
    sentiment_threshold_bullish: float = 0.6
    sentiment_threshold_bearish: float = -0.6
    poll_interval_seconds: int = 60  # how often to fetch fresh headlines


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters — INR based."""
    max_daily_drawdown_pct: float = 0.03          # 3 % circuit breaker
    default_stop_loss_pct: float = 0.02            # 2 % stop-loss
    default_take_profit_pct: float = 0.04          # 4 % take-profit
    kelly_fraction_cap: float = 0.25               # never exceed 25 % of equity
    max_position_pct: float = 0.10                 # single position ≤ 10 % of equity
    max_open_positions: int = 10
    slippage_bps: float = 5.0                      # 5 basis points assumed slippage
    commission_per_share: float = 0.0              # adjust for your broker


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy hyper-parameters."""
    # Momentum / MA crossover
    momentum_fast_window: int = 10
    momentum_slow_window: int = 50
    # Mean reversion
    mean_rev_lookback: int = 20
    mean_rev_z_entry: float = 2.0
    mean_rev_z_exit: float = 0.5
    # Sentiment
    sentiment_extreme_score: float = 0.75  # absolute score threshold


@dataclass(frozen=True)
class SystemConfig:
    """Top-level system knobs — Indian markets only."""
    # NSE large-cap equities
    nse_tickers: List[str] = field(
        default_factory=lambda: [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "HINDUNILVR", "WIPRO",
            "LT", "AXISBANK", "KOTAKBANK", "MARUTI", "TATAMOTORS",
            "ADANIENT", "BAJFINANCE", "SUNPHARMA", "TITAN", "NESTLEIND",
        ]
    )
    # MCX commodities
    mcx_tickers: List[str] = field(
        default_factory=lambda: [
            "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER",
        ]
    )
    # Combined ticker list used everywhere
    tickers: List[str] = field(default_factory=list)
    currency: str = "INR"
    currency_symbol: str = "₹"
    exchange: str = "NSE"  # primary exchange
    heartbeat_interval_sec: int = 10
    dashboard_port: int = 8501
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    data_provider: str = os.getenv("DATA_PROVIDER", "yfinance")  # "yfinance" | "synthetic"
    # IST market hours: 09:15 – 15:30
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30

    def __post_init__(self):
        # Populate combined tickers if empty (frozen dataclass workaround)
        if not self.tickers:
            combined = list(self.nse_tickers) + list(self.mcx_tickers)
            object.__setattr__(self, "tickers", combined)


# ── Singleton instances ──────────────────────────────────────────────
NLP = NLPConfig()
RISK = RiskConfig()
STRATEGY = StrategyConfig()
SYSTEM = SystemConfig()
