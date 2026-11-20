"""
Professional Trading Terminal — Indian Markets (NSE / BSE / MCX).

Bloomberg-grade dashboard with:
  1. Live RSS news from 13+ Indian financial sources
  2. Full NSE/BSE/MCX stock universe (300+ stocks)
  3. Interactive candlestick charts with drawing tools
  4. Technical indicators (RSI, MACD, Bollinger, Ichimoku, Supertrend, etc.)
  5. Trade entry/exit markers with P&L annotations
  6. Fibonacci retracement & pivot point overlays
  7. Professional P&L analytics, equity curve, strategy comparison
  8. Real-time sentiment feed with NLP scoring
  9. Multi-timeframe analysis

Run with:
    streamlit run algo_trading/dashboard/app.py
"""

from __future__ import annotations

import sys
import time
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Ensure the project root is on sys.path ───────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import SYSTEM, RISK
from core.state_store import read_state, is_live
from analytics.trade_analytics import TradeAnalytics
from analytics.technical_indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic,
    adx, supertrend, ichimoku, vwap, obv, pivot_points,
    fibonacci_retracement, detect_patterns, parabolic_sar,
)
from data_engine.news_feed import (
    fetch_all_news, news_to_dataframe, ALL_STOCKS,
    NIFTY_50, NIFTY_NEXT_50, MCX_COMMODITIES, get_yfinance_symbol,
)
from analytics.options_engine import (
    OptionsEngine, OptionLeg, OptionType, OptionAction,
)
from analytics.market_regime import MarketRegimeDetector, MarketRegime
from analytics.session_manager import get_session_info, IndianSession
from analytics.monte_carlo import simulate_equity_paths
from analytics.portfolio_risk import compute_correlation_matrix, compute_portfolio_metrics
from analytics.candle_maker import (
    compute_volume_profile, compute_heikin_ashi, compute_anchored_vwap_bands,
)
from data_engine.corporate_actions import CorporateActionsPipeline
from analytics.trade_recommender import TradeRecommender

# ═══════════════════════════════════════════════════════════════════════
#  Page config — dark professional theme
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AlgoTerminal Pro — NSE/BSE/MCX",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional dark CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ═════════════════════════════════════════════════════════════════
       GLOBAL THEME — Bloomberg Terminal Inspired
       ═════════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary:    #000000;
        --bg-secondary:  #080808;
        --bg-card:       #0a0a0a;
        --bg-elevated:   #111111;
        --border-subtle: #1c1c1c;
        --border-accent: #333333;
        --text-primary:  #f0f0f0;
        --text-secondary:#a0a0a0;
        --text-muted:    #666666;
        --accent-blue:   #ffffff;
        --accent-green:  #00ff41;
        --accent-red:    #ff3333;
        --accent-orange: #a0a0a0;
        --accent-purple: #a0a0a0;
        --accent-pink:   #a0a0a0;
        --accent-cyan:   #a0a0a0;
        --accent-gold:   #a0a0a0;
        --shadow-sm:     none;
        --shadow-md:     none;
        --shadow-lg:     none;
        --shadow-glow:   0 0 10px rgba(255,255,255,0.05);
        --radius-sm:     2px;
        --radius-md:     4px;
        --radius-lg:     6px;
    }

    /* ── App Container ────────────────────────────────────────────── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #050505 !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: var(--text-primary);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.92rem !important;
        padding: 6px 10px !important;
        border-radius: var(--radius-sm) !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: var(--bg-elevated) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── Header Banner ────────────────────────────────────────────── */
    .pro-header {
        background: #0a0a0a;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 16px 24px;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }
    .pro-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--border-accent);
    }
    .pro-header h1 {
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        text-shadow: none;
    }
    .pro-header p {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin: 4px 0 0;
    }

    /* ── Metric Cards — No Truncation Across Any Page ────────── */
    div[data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        padding: 8px 10px !important;
        transition: all 0.2s ease;
        overflow: visible !important;
        min-width: 0 !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-accent) !important;
        background: var(--bg-elevated) !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] p {
        color: var(--text-muted) !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.4px !important;
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] * {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: clamp(0.85rem, 1.15vw, 1.20rem) !important;
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
        letter-spacing: -0.3px !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] * {
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
        font-size: 0.72rem !important;
    }

    /* ── Ticker Tape ──────────────────────────────────────────────── */
    .ticker-tape {
        background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-card) 50%, var(--bg-secondary) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 10px 20px;
        overflow-x: auto;
        white-space: nowrap;
        font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
        font-size: 0.82rem;
        box-shadow: var(--shadow-sm);
        scrollbar-width: thin;
        scrollbar-color: var(--border-accent) transparent;
    }
    .ticker-tape::-webkit-scrollbar { height: 4px; }
    .ticker-tape::-webkit-scrollbar-track { background: transparent; }
    .ticker-tape::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 2px; }
    .tk-up   { color: var(--accent-green); font-weight: 700; text-shadow: 0 0 8px rgba(63,185,80,0.3); }
    .tk-down { color: var(--accent-red);   font-weight: 700; text-shadow: 0 0 8px rgba(248,81,73,0.3); }
    .tk-name { color: var(--text-secondary); font-weight: 500; }

    /* ── Section Headers ──────────────────────────────────────────── */
    .section-hdr {
        border-left: 2px solid var(--border-accent);
        padding-left: 10px;
        margin: 24px 0 12px 0;
    }
    .section-hdr h4 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: -0.2px;
        margin: 0 !important;
    }

    /* ── News Badges ──────────────────────────────────────────────── */
    .badge-critical {
        background: linear-gradient(135deg, #da3633, #b62324);
        color: white; padding: 3px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(218,54,51,0.3);
    }
    .badge-high {
        background: linear-gradient(135deg, #f0883e, #d87030);
        color: white; padding: 3px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-normal {
        background: linear-gradient(135deg, #388bfd, #2563eb);
        color: white; padding: 3px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 500;
    }

    /* ── News Card ────────────────────────────────────────────────── */
    .news-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .news-card:hover {
        border-color: var(--accent-blue);
        box-shadow: var(--shadow-md);
        transform: translateX(3px);
    }
    .news-card .headline {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.4;
    }
    .news-card .meta {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 2px;
    }
    .news-card .desc {
        color: var(--text-secondary);
        font-size: 0.82rem;
        margin-top: 6px;
        line-height: 1.4;
    }

    /* ── Drawing Toolbar ──────────────────────────────────────────── */
    .draw-toolbar {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius-sm);
        padding: 8px 16px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    .draw-toolbar b { color: var(--accent-cyan); }

    /* ── Strategy Insight Cards ───────────────────────────────────── */
    .insight-card {
        border-radius: var(--radius-md);
        padding: 16px 18px;
        box-shadow: var(--shadow-sm);
        transition: all 0.25s ease;
    }
    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .insight-green  { background: linear-gradient(145deg, #0d2818, #0f3020); border: 1px solid #238636; }
    .insight-blue   { background: linear-gradient(145deg, #0d1b2e, #0f2040); border: 1px solid #1f6feb; }
    .insight-red    { background: linear-gradient(145deg, #2d1117, #36151c); border: 1px solid #f85149; }

    /* ── DataFrames & Tables ──────────────────────────────────────── */
    .stDataFrame, div[data-testid="stDataFrame"] {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }
    div[data-testid="stDataFrame"] > div {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
    }

    /* ── Buttons & Inputs ─────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
    }
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
    }

    /* ── Scrollbar Global ─────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ── Status Badges (Live/Demo) ────────────────────────────────── */
    div[data-testid="stAlert"] {
        border-radius: var(--radius-sm) !important;
        font-size: 0.85rem !important;
    }

    /* ── Dividers ─────────────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-subtle) !important;
        margin: 12px 0 !important;
    }

    /* ── Plotly Charts Container ──────────────────────────────────── */
    .stPlotlyChart {
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    /* ── Top Navigation Bar Buttons ──────────────────────────────── */
    div[data-testid="column"] button {
        border-radius: var(--radius-sm) !important;
        padding: 6px 4px !important;
        font-size: 0.78rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="column"] button[kind="primary"] {
        background-color: #111111 !important;
        border: 1px solid #00ff41 !important;
        color: #00ff41 !important;
        font-weight: 700 !important;
    }
    div[data-testid="column"] button[kind="secondary"] {
        background-color: #080808 !important;
        border: 1px solid #1c1c1c !important;
        color: #777777 !important;
        font-weight: 500 !important;
    }
    div[data-testid="column"] button[kind="secondary"]:hover {
        border-color: #333333 !important;
        color: #ffffff !important;
        background-color: #111111 !important;
    }

    /* ── Sidebar Toggle Button High Contrast ──────────────────────── */
    button[data-testid="collapsedControl"] {
        display: flex !important;
        background: #111111 !important;
        border: 1px solid #333333 !important;
        color: #00ff41 !important;
        border-radius: 4px !important;
    }

    /* ── Smooth Animations ────────────────────────────────────────── */
    * { transition-property: background-color, border-color, box-shadow, transform, opacity;
        transition-duration: 0.15s;
        transition-timing-function: ease-out; }

    /* ── Expander ─────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
    }

    /* ── Toggle Switch ────────────────────────────────────────────── */
    .stToggle label span {
        font-size: 0.88rem !important;
    }

    /* ── Page Footer ──────────────────────────────────────────────── */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

CUR = SYSTEM.currency_symbol  # ₹


# ═══════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════

def _load_live_state() -> dict | None:
    """Attempt to read state written by the live trading engine."""
    raw = read_state()
    if raw is None:
        return None
    try:
        snap = raw["snapshot"]
        positions = pd.DataFrame(raw.get("positions") or [])
        trade_log = pd.DataFrame(raw.get("trade_log") or [])
        sentiment_feed = pd.DataFrame(raw.get("sentiment_feed") or [])

        eq_records = raw.get("equity_curve") or []
        if eq_records:
            equity_curve = pd.DataFrame(eq_records)
            equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], format="mixed", utc=True)
        else:
            equity_curve = pd.DataFrame(columns=["timestamp", "equity"])

        bars: dict[str, pd.DataFrame] = {}
        for ticker, bar_list in (raw.get("bars") or {}).items():
            df = pd.DataFrame(bar_list)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
            bars[ticker] = df

        return {
            "snapshot": snap,
            "positions": positions,
            "bars": bars,
            "equity_curve": equity_curve,
            "sentiment_feed": sentiment_feed,
            "trade_log": trade_log,
            "closed_trades": raw.get("closed_trades") or [],
            "prev_closes": raw.get("prev_closes") or {},
            "dividends": raw.get("dividends") or [],
            "options_chains": raw.get("options_chains") or {},
            "regime_info": raw.get("regime_info") or {},
            "session_info": raw.get("session_info") or {},
            "forming_candles": raw.get("forming_candles") or {},
            "trade_suggestions": raw.get("trade_suggestions") or [],
        }
    except (KeyError, TypeError):
        return None


def _generate_demo_data() -> dict:
    """Generate realistic demo data for Indian markets."""
    np.random.seed(42)
    now = datetime.now(timezone.utc)

    # Use the full stock universe for demo
    from data_engine.historical_loader import _INR_BASE_PRICES
    tickers = SYSTEM.tickers

    bars: dict[str, pd.DataFrame] = {}
    for t in tickers:
        n = 200
        base = _INR_BASE_PRICES.get(t, 1000 + np.random.randn() * 500)
        close = base + np.cumsum(np.random.randn(n) * base * 0.003)
        high = close + np.abs(np.random.randn(n) * base * 0.002)
        low = close - np.abs(np.random.randn(n) * base * 0.002)
        opn = close + np.random.randn(n) * base * 0.001
        vol = np.random.randint(1000, 50000, n).astype(float)
        ts = pd.date_range(end=now, periods=n, freq="1min", tz="UTC")
        bars[t] = pd.DataFrame({
            "timestamp": ts, "open": opn, "high": high,
            "low": low, "close": close, "volume": vol,
        })

    positions = pd.DataFrame([
        {"Ticker": "RELIANCE", "Side": "BUY", "Qty": 50, "Entry": 2940.50,
         "Current": 2965.30, "Unrealized PnL": 1240.00, "Stop Loss": 2881.69,
         "Take Profit": 3058.12, "Strategy": "momentum_ma_crossover"},
        {"Ticker": "TCS", "Side": "BUY", "Qty": 20, "Entry": 4180.00,
         "Current": 4215.80, "Unrealized PnL": 716.00, "Stop Loss": 4096.40,
         "Take Profit": 4347.20, "Strategy": "mean_reversion_zscore"},
        {"Ticker": "GOLD", "Side": "BUY", "Qty": 5, "Entry": 72200.00,
         "Current": 72680.00, "Unrealized PnL": 2400.00, "Stop Loss": 70756.00,
         "Take Profit": 75088.00, "Strategy": "momentum_ma_crossover"},
    ])

    eq_n = 500
    eq = 10_00_000 + np.cumsum(np.random.randn(eq_n) * 500)
    equity_curve = pd.DataFrame({
        "timestamp": pd.date_range(end=now, periods=eq_n, freq="1min", tz="UTC"),
        "equity": eq,
    })

    demo_trades = [
        {"ticker": "RELIANCE", "side": "buy", "entry": 2920.00, "exit": 2965.30, "qty": 50, "pnl": 2265.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T04:15:00"},
        {"ticker": "TCS", "side": "buy", "entry": 4150.00, "exit": 4215.80, "qty": 20, "pnl": 1316.00, "strategy": "sentiment_extreme", "closed_at": "2026-04-01T04:45:00"},
        {"ticker": "INFY", "side": "buy", "entry": 1860.00, "exit": 1842.50, "qty": 60, "pnl": -1050.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T05:15:00"},
        {"ticker": "HDFCBANK", "side": "buy", "entry": 1630.00, "exit": 1662.30, "qty": 40, "pnl": 1292.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T05:30:00"},
        {"ticker": "ICICIBANK", "side": "buy", "entry": 1240.00, "exit": 1258.50, "qty": 50, "pnl": 925.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T05:45:00"},
        {"ticker": "SBIN", "side": "buy", "entry": 815.00, "exit": 828.50, "qty": 100, "pnl": 1350.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T06:00:00"},
        {"ticker": "BHARTIARTL", "side": "buy", "entry": 1690.00, "exit": 1672.00, "qty": 30, "pnl": -540.00, "strategy": "sentiment_extreme", "closed_at": "2026-04-01T06:15:00"},
        {"ticker": "ITC", "side": "buy", "entry": 465.00, "exit": 472.50, "qty": 150, "pnl": 1125.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T06:30:00"},
        {"ticker": "TATAMOTORS", "side": "buy", "entry": 790.00, "exit": 778.50, "qty": 80, "pnl": -920.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T06:45:00"},
        {"ticker": "GOLD", "side": "buy", "entry": 72000.00, "exit": 72680.00, "qty": 5, "pnl": 3400.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T07:45:00"},
        {"ticker": "SILVER", "side": "buy", "entry": 87500.00, "exit": 86800.00, "qty": 3, "pnl": -2100.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T08:00:00"},
        {"ticker": "BAJFINANCE", "side": "buy", "entry": 7350.00, "exit": 7420.00, "qty": 10, "pnl": 700.00, "strategy": "sentiment_extreme", "closed_at": "2026-04-01T08:30:00"},
        {"ticker": "SUNPHARMA", "side": "buy", "entry": 1730.00, "exit": 1762.00, "qty": 30, "pnl": 960.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T09:00:00"},
        {"ticker": "KOTAKBANK", "side": "buy", "entry": 1880.00, "exit": 1912.00, "qty": 25, "pnl": 800.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T09:30:00"},
        {"ticker": "NESTLEIND", "side": "buy", "entry": 2330.00, "exit": 2365.00, "qty": 20, "pnl": 700.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T09:45:00"},
    ]

    analytics = TradeAnalytics(trades=demo_trades)

    snapshot = {
        "equity": float(eq[-1]),
        "cash": float(eq[-1] * 0.55),
        "daily_pnl": float(eq[-1] - 10_00_000),
        "daily_pnl_pct": float((eq[-1] - 10_00_000) / 10_00_000 * 100),
        "drawdown_pct": 0.8,
        "peak_equity": float(max(eq)),
        "open_positions": 3,
        "halted": False,
    }

    trade_log = pd.DataFrame(demo_trades)
    if "closed_at" in trade_log.columns:
        trade_log["Time"] = pd.to_datetime(trade_log["closed_at"], utc=True).dt.strftime("%H:%M")

    sentiment_feed = pd.DataFrame([
        {"Time": "09:20", "Ticker": "RELIANCE", "Headline": "Reliance Jio 5G rollout accelerates", "Source": "ET", "Sentiment": "bullish", "Score": 0.85},
        {"Time": "09:35", "Ticker": "TCS", "Headline": "TCS bags $2B deal from European bank", "Source": "Moneycontrol", "Sentiment": "bullish", "Score": 0.88},
        {"Time": "09:50", "Ticker": "INFY", "Headline": "Infosys revises guidance downward", "Source": "NDTV Profit", "Sentiment": "bearish", "Score": -0.62},
        {"Time": "10:05", "Ticker": "HDFCBANK", "Headline": "HDFC Bank NPA ratio hits record low", "Source": "LiveMint", "Sentiment": "bullish", "Score": 0.79},
        {"Time": "10:20", "Ticker": "GOLD", "Headline": "Gold hits fresh all-time high", "Source": "Reuters", "Sentiment": "bullish", "Score": 0.91},
        {"Time": "10:35", "Ticker": "TATAMOTORS", "Headline": "Tata Motors JLR margins under pressure", "Source": "Bloomberg Quint", "Sentiment": "bearish", "Score": -0.55},
    ])

    return {
        "snapshot": snapshot,
        "positions": positions,
        "bars": bars,
        "equity_curve": equity_curve,
        "sentiment_feed": sentiment_feed,
        "trade_log": trade_log,
        "analytics": analytics.summary(),
        "analytics_by_strategy": analytics.by_strategy(),
        "analytics_by_ticker": analytics.by_ticker(),
        "trades_df": analytics.to_dataframe(),
        "pnl_by_ticker": analytics.pnl_by_ticker_df(),
        "closed_trades": demo_trades,
        "options_chains": {},
        "regime_info": {
            "regime": "TRENDING_UP",
            "adx_value": 28.5,
            "atr_pct": 1.42,
            "bb_width": 0.048,
            "squeeze": False,
            "ticker": "RELIANCE",
        },
        "session_info": {
            "session": "power_hour",
            "label": "Power Hour",
            "start_time": "14:30",
            "end_time": "15:30",
            "minutes_remaining": 45,
            "should_avoid_entry": False,
            "is_orb_window": False,
        },
        "forming_candles": {
            "RELIANCE": {
                "open": 2960.0, "high": 2968.5, "low": 2959.0, "close": 2965.3,
                "volume": 8420.0, "buy_volume": 5200.0, "sell_volume": 3220.0,
                "delta": 1980.0, "tick_count": 86, "open_time": now.isoformat(),
            }
        },
        "dividends": CorporateActionsPipeline().dividend_results,
        "trade_suggestions": TradeRecommender().generate_recommendations(
            bars_dict=bars,
            options_chains={
                "NIFTY 50": {
                    "underlying": "NIFTY 50",
                    "spot_price": 24500.0,
                    "atm_strike": 24500.0,
                    "pcr_oi": 1.28,
                    "max_pain_strike": 24400.0,
                    "iv_rank": 48.0,
                    "strikes": [
                        {"strike": 24500.0, "call_ltp": 175.0, "call_delta": 0.52, "call_gamma": 0.0018, "call_theta": -12.4, "call_vega": 18.2, "put_ltp": 160.0, "put_delta": -0.48},
                        {"strike": 24550.0, "call_ltp": 142.0, "call_delta": 0.45, "call_gamma": 0.0019, "call_theta": -11.8, "call_vega": 17.5, "put_ltp": 195.0, "put_delta": -0.55},
                        {"strike": 24400.0, "call_ltp": 235.0, "call_delta": 0.62, "call_gamma": 0.0016, "call_theta": -13.0, "call_vega": 19.0, "put_ltp": 115.0, "put_delta": -0.38},
                    ]
                },
                "BANKNIFTY": {
                    "underlying": "BANKNIFTY",
                    "spot_price": 52100.0,
                    "atm_strike": 52100.0,
                    "pcr_oi": 1.05,
                    "max_pain_strike": 52000.0,
                    "iv_rank": 52.0,
                    "strikes": [
                        {"strike": 52100.0, "call_ltp": 320.0, "call_delta": 0.50, "call_gamma": 0.0012, "call_theta": -22.0, "call_vega": 34.0, "put_ltp": 310.0, "put_delta": -0.50},
                    ]
                }
            },
            regime_info={"regime": "TRENDING_UP", "adx_value": 28.5},
            prev_closes={t: bars[t]["close"].iloc[-2] for t in tickers if not bars[t].empty},
            sentiment_scores={"RELIANCE": 0.85, "TCS": 0.88, "INFY": -0.62, "HDFCBANK": 0.79},
            account_equity=snapshot["equity"],
        ),
    }


def _load_data() -> dict:
    live = _load_live_state()
    if live is not None:
        closed = live.get("closed_trades") or []
        if closed:
            an = TradeAnalytics(trades=closed)
            live["analytics"] = an.summary()
            live["analytics_by_strategy"] = an.by_strategy()
            live["analytics_by_ticker"] = an.by_ticker()
            live["trades_df"] = an.to_dataframe()
            live["pnl_by_ticker"] = an.pnl_by_ticker_df()
        else:
            live.setdefault("analytics", {})
            live.setdefault("analytics_by_strategy", {})
            live.setdefault("analytics_by_ticker", {})
            live.setdefault("trades_df", pd.DataFrame())
            live.setdefault("pnl_by_ticker", pd.DataFrame())

        divs = live.get("dividends")
        if not divs or isinstance(divs, list) or (isinstance(divs, dict) and not divs.get("upcoming")):
            ca = CorporateActionsPipeline()
            ca._build_upcoming_schedule()
            live["dividends"] = ca.dividend_results

        if not live.get("trade_suggestions"):
            live["trade_suggestions"] = TradeRecommender().generate_recommendations(
                bars_dict=live.get("bars", {}),
                options_chains=live.get("options_chains", {}),
                regime_info=live.get("regime_info", {}),
                prev_closes=live.get("prev_closes", {}),
                account_equity=live.get("snapshot", {}).get("equity", 10_00_000.0),
            )
        return live
    return _generate_demo_data()


data = _load_data()
live_mode = is_live()
snap = data["snapshot"]
stats = data.get("analytics") or {}


PAGES = [
    "Terminal", "Trade Suggestions", "Dividends Desk", "Options Desk",
    "Chart Lab", "News Room", "Risk Lab", "P&L Analytics", "Strategy Hub", "Trade Log"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Terminal"
elif st.session_state["current_page"] == "Dividends":
    st.session_state["current_page"] = "Dividends Desk"

# ═══════════════════════════════════════════════════════════════════════
#  Sidebar — Professional Controls
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### AlgoTerminal Pro")
    st.caption("Institutional-Grade Trading System")
    st.divider()

    # Sidebar Navigation
    st.markdown("**Navigation Menu**")
    for p_name in PAGES:
        is_active = (st.session_state["current_page"] == p_name)
        if st.button(p_name, key=f"sb_nav_btn_{p_name}", use_container_width=True, type="primary" if is_active else "secondary"):
            if st.session_state["current_page"] != p_name:
                st.session_state["current_page"] = p_name
                st.rerun()
    st.divider()

    # Stock universe selector
    universe = st.selectbox("Stock Universe", [
        "Trading Portfolio (25)", "NIFTY 50", "NIFTY Next 50",
        "MCX Commodities", "Full Universe (300+)",
    ])

    if universe == "Trading Portfolio (25)":
        stock_list = SYSTEM.tickers
    elif universe == "NIFTY 50":
        stock_list = NIFTY_50
    elif universe == "NIFTY Next 50":
        stock_list = NIFTY_NEXT_50
    elif universe == "MCX Commodities":
        stock_list = MCX_COMMODITIES
    else:
        stock_list = ALL_STOCKS

    selected_ticker = st.selectbox("Select Stock", stock_list)
    st.divider()
    
    # Dynamic Add Stock
    st.markdown("### Add Stock to Tracker")
    new_ticker = st.text_input("NSE Symbol (e.g. TATASTEEL)", key="add_ticker")
    if st.button("Add Ticker", use_container_width=True):
        if new_ticker:
            tk_upper = new_ticker.strip().upper()
            try:
                import os
                txt_path = os.path.join("data", "dynamic_tickers.txt")
                os.makedirs("data", exist_ok=True)
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write(tk_upper + "\n")
                st.success(f"Added {tk_upper} to tracking queue")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    
    # Manual Trade Entry
    st.markdown("### Log Manual Trade")
    with st.form("manual_trade_form", clear_on_submit=True):
        col_t1, col_t2 = st.columns(2)
        m_ticker = col_t1.text_input("Ticker", placeholder="INFY")
        m_side = col_t2.selectbox("Side", ["BUY", "SELL"])
        col_t3, col_t4 = st.columns(2)
        m_qty = col_t3.number_input("Qty", min_value=1, value=10)
        m_entry = col_t4.number_input("Entry Price", min_value=0.0, value=100.0, format="%.2f")
        m_exit = st.number_input("Exit Price (optional)", min_value=0.0, value=0.0, format="%.2f")
        if st.form_submit_button("Save Trade", use_container_width=True):
            if m_ticker:
                try:
                    import json, os, uuid, datetime
                    trade_path = os.path.join("data", "manual_trades.json")
                    os.makedirs("data", exist_ok=True)
                    m_trade = {
                        "id": str(uuid.uuid4())[:8],
                        "ticker": m_ticker.upper(),
                        "side": m_side.lower(),
                        "qty": m_qty,
                        "entry": m_entry,
                        "exit": m_exit if m_exit > 0 else m_entry,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "strategy": "manual"
                    }
                    existing = []
                    if os.path.exists(trade_path):
                        with open(trade_path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                    existing.append(m_trade)
                    with open(trade_path, "w", encoding="utf-8") as f:
                        json.dump(existing, f)
                    st.success(f"Trade saved for {m_ticker.upper()}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    if live_mode:
        st.success("[LIVE ENGINE] 1s Refresh")
    else:
        st.warning("[DEMO MODE]")

    if snap.get("halted"):
        st.error("[CIRCUIT BREAKER]")
    else:
        st.success("[TRADING ACTIVE]")

    st.divider()
    st.caption(f"Drawdown limit: {RISK.max_daily_drawdown_pct:.0%}")
    st.caption(f"Max positions: {RISK.max_open_positions}")
    st.caption(f"Stop-loss: {RISK.default_stop_loss_pct:.0%}")
    st.caption(f"Take-profit: {RISK.default_take_profit_pct:.0%}")
    st.caption(f"Universe size: {len(stock_list)} stocks")


# ── Institutional Top Navigation Bar (Always Visible Across All Screens) ──
nav_cols = st.columns(len(PAGES))
for idx, p_name in enumerate(PAGES):
    is_active = (st.session_state["current_page"] == p_name)
    btn_type = "primary" if is_active else "secondary"
    if nav_cols[idx].button(p_name, key=f"top_nav_btn_{p_name}", use_container_width=True, type=btn_type):
        if st.session_state["current_page"] != p_name:
            st.session_state["current_page"] = p_name
            st.rerun()

page = st.session_state["current_page"]
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1: TERMINAL (Main Dashboard)
# ═══════════════════════════════════════════════════════════════════════
if page == "Terminal":
    st.markdown("""
    <div class="pro-header">
        <h1>ALGO TRADING TERMINAL</h1>
        <p>Real-Time Execution & Analytics Engine</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Executive Quick Navigation Buttons (Prominent Action Bar) ──
    eq_c1, eq_c2, eq_c3, eq_c4 = st.columns(4)
    if eq_c1.button("TRADE SUGGESTIONS & OPTION PLAYS", key="top_quick_trades", use_container_width=True, type="primary"):
        st.session_state["current_page"] = "Trade Suggestions"
        st.rerun()
    if eq_c2.button("UPCOMING DIVIDENDS & EXIT RADAR", key="top_quick_divs", use_container_width=True):
        st.session_state["current_page"] = "Dividends Desk"
        st.rerun()
    if eq_c3.button("DERIVATIVES & OPTIONS DESK", key="top_quick_options", use_container_width=True):
        st.session_state["current_page"] = "Options Desk"
        st.rerun()
    if eq_c4.button("INSTITUTIONAL CHART LAB", key="top_quick_chart", use_container_width=True):
        st.session_state["current_page"] = "Chart Lab"
        st.rerun()

    # ── Ticker Tape ───────────────────────────────────────────────────
    tape_items = []
    for tk in SYSTEM.tickers[:15]:
        bdf = data["bars"].get(tk, pd.DataFrame())
        if not bdf.empty:
            last = bdf["close"].iloc[-1]
            last = bdf["close"].iloc[-1]
            prev_close = data["prev_closes"].get(tk, last)
            if prev_close == 0:
                prev_close = last
            
            chg = last - prev_close
            chg_pct = (chg / prev_close * 100) if prev_close != 0 else 0
            cls = "tk-up" if chg >= 0 else "tk-down"
            arrow = "▲" if chg >= 0 else "▼"
            tape_items.append(
                f'<span class="tk-name">{tk}</span> '
                f'<span class="{cls}">₹{last:,.1f} {arrow}{abs(chg_pct):.1f}%</span>'
            )
    if tape_items:
        st.markdown(
            f'<div class="ticker-tape">{" &nbsp;│&nbsp; ".join(tape_items)}</div>',
            unsafe_allow_html=True,
        )

    # ── Market Regime & Indian Session Status Banner ───────────────────
    regime_data = data.get("regime_info") or {}
    regime_name = str(regime_data.get("regime", "RANGING")).upper()
    adx_val = float(regime_data.get("adx_value", 21.0))
    atr_val = float(regime_data.get("atr_pct", 1.25))
    bb_val = float(regime_data.get("bb_width", 0.045))

    regime_colors = {
        "TRENDING_UP": "#00ff41",
        "TRENDING_DOWN": "#ff3333",
        "RANGING": "#e6b800",
        "HIGH_VOLATILITY": "#ff6600",
        "SQUEEZE": "#00d4ff",
    }
    r_color = regime_colors.get(regime_name, "#a0a0a0")

    sess_live = get_session_info()
    sess_label = sess_live.get("label", "Market Closed")
    mins_rem = sess_live.get("minutes_remaining", 0)
    avoid_entry = sess_live.get("should_avoid_entry", False)
    is_orb = sess_live.get("is_orb_window", False)

    sess_badge_color = "#ff3333" if avoid_entry else ("#00d4ff" if is_orb else "#00ff41")
    sess_status_note = "LUNCH DOLDRUMS (ENTRY PAUSED)" if avoid_entry else ("ORB 15-MIN WINDOW" if is_orb else f"{mins_rem}m remaining")

    st.markdown(f"""
    <div style="display: flex; gap: 12px; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;">
        <div style="flex: 1; background: #0a0a0a; border: 1px solid #1c1c1c; border-left: 4px solid {r_color}; padding: 8px 14px; border-radius: 4px;">
            <div style="font-size: 11px; color: #666666; text-transform: uppercase; letter-spacing: 0.5px;">Market Regime Detection</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px;">
                <span style="font-size: 15px; font-weight: 700; color: {r_color};">{regime_name.replace('_', ' ')}</span>
                <span style="font-size: 11px; color: #a0a0a0;">ADX: <b>{adx_val:.1f}</b> │ ATR: <b>{atr_val:.2f}%</b> │ BB Width: <b>{bb_val:.3f}</b></span>
            </div>
        </div>
        <div style="flex: 1; background: #0a0a0a; border: 1px solid #1c1c1c; border-left: 4px solid {sess_badge_color}; padding: 8px 14px; border-radius: 4px;">
            <div style="font-size: 11px; color: #666666; text-transform: uppercase; letter-spacing: 0.5px;">NSE/BSE Session Clock (IST)</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px;">
                <span style="font-size: 15px; font-weight: 700; color: #ffffff;">{sess_label}</span>
                <span style="font-size: 11px; font-weight: 600; color: {sess_badge_color};">{sess_status_note}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Portfolio Value", f"₹{snap['equity']:,.0f}")
    c2.metric("Daily P&L", f"₹{snap['daily_pnl']:+,.0f}", delta=f"{snap['daily_pnl_pct']:.2f}%")
    c3.metric("Cash", f"₹{snap['cash']:,.0f}")
    c4.metric("Drawdown", f"{snap.get('drawdown_pct', 0):.2f}%")
    c5.metric("Open Pos.", snap.get("open_positions", 0))
    win_rate = stats.get("win_rate_pct", 0)
    c6.metric("Win Rate", f"{win_rate:.1f}%")

    # ── Equity + Quick Chart ──────────────────────────────────────────
    col_eq, col_chart = st.columns([1, 2])

    with col_eq:
        st.markdown('<div class="section-hdr"><h4>Equity Curve</h4></div>', unsafe_allow_html=True)
        eq_df = data["equity_curve"].copy()
        if not eq_df.empty:
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], format="mixed", utc=True)
            eq_df["timestamp"] = eq_df["timestamp"].dt.tz_convert('Asia/Kolkata')

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["timestamp"], y=eq_df["equity"],
                mode="lines", line=dict(color="#3fb950", width=2),
                fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
                hovertemplate="₹%{y:,.0f}<extra></extra>",
            ))
            fig_eq.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="INR", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

    with col_chart:
        # Title and Timeframe Selector side-by-side
        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            st.markdown(f'<div class="section-hdr"><h4>Quick View — {selected_ticker}</h4></div>', unsafe_allow_html=True)
        with q_col2:
            tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=0, key="quick_tf", label_visibility="collapsed")
            
        bar_df = data["bars"].get(selected_ticker, pd.DataFrame())
        if not bar_df.empty:
            plot_df = bar_df.copy()
            plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], format="mixed", utc=True)
            plot_df["timestamp"] = plot_df["timestamp"].dt.tz_convert('Asia/Kolkata')
            
            # Resample if needed
            if tf != "1m":
                rule_map = {"1m": "1T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "1H", "1d": "1D"}
                plot_df.set_index("timestamp", inplace=True)
                plot_df = plot_df.resample(rule_map[tf]).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                }).dropna()
                plot_df.reset_index(inplace=True)
                
            fig_q = go.Figure(data=[go.Candlestick(
                x=plot_df["timestamp"],
                open=plot_df["open"], high=plot_df["high"],
                low=plot_df["low"], close=plot_df["close"],
                increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            )])
            fig_q.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info(f"No data for {selected_ticker}")

    # ── Active Positions ──────────────────────────────────────────────
    st.markdown('<div class="section-hdr"><h4>Active Positions</h4></div>', unsafe_allow_html=True)
    pos_df = data["positions"]
    if not pos_df.empty:
        pnl_col = "Unrealized PnL" if "Unrealized PnL" in pos_df.columns else None
        if pnl_col:
            def _color_pnl(val):
                if isinstance(val, (int, float)):
                    return f"color: {'#3fb950' if val >= 0 else '#f85149'}; font-weight: bold"
                return ""
            fmt = {c: "₹{:.2f}" for c in ["Entry", "Current", "Unrealized PnL", "Stop Loss", "Take Profit"] if c in pos_df.columns}
            st.dataframe(pos_df.style.map(_color_pnl, subset=[pnl_col]).format(fmt),
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

    # ── Institutional Trade Recommendations & Option Plays ────────────
    st.markdown('<div class="section-hdr"><h4>Institutional Trade Recommendations & Option Plays</h4></div>', unsafe_allow_html=True)
    suggestions = data.get("trade_suggestions") or []
    if suggestions:
        sug_cols = st.columns(min(len(suggestions), 4))
        for idx, sug in enumerate(suggestions[:4]):
            with sug_cols[idx]:
                action     = sug.get("action_label", "BUY")
                t_type     = sug.get("trade_type", "EQUITY")
                direction  = sug.get("direction", "BULLISH")
                is_neutral = direction == "NEUTRAL" or "CONDOR" in str(t_type) or "STRANGLE" in str(t_type)
                is_bull    = (not is_neutral) and (direction == "BULLISH" or ("BUY" in action and "PUT" not in action))
                is_bear    = (not is_neutral) and (direction == "BEARISH" or "PUT" in action or "SELL" in action)
                badge_bg   = "#ffd700" if is_neutral else ("#00ff41" if is_bull else "#ff3333")
                badge_txt  = "#000000" if (is_neutral or is_bull) else "#ffffff"
                dir_clr    = "#ffd700" if is_neutral else ("#00ff41" if is_bull else "#ff3333")

                inst_name   = sug.get("instrument_name", sug.get("ticker"))
                expiry_lbl  = sug.get("expiry_label", "")
                entry_px    = sug.get("entry_price", 0.0)
                ez_low      = sug.get("entry_zone_low", entry_px * 0.998)
                ez_high     = sug.get("entry_zone_high", entry_px * 1.002)
                tgt1        = sug.get("target_1", 0.0)
                tgt2        = sug.get("target_2", 0.0)
                tgt3        = sug.get("target_3", 0.0)
                sl          = sug.get("stop_loss", 0.0)
                rr          = sug.get("risk_reward", "1 : 2.0")
                conf        = sug.get("confidence_pct", 80)
                score_total = sug.get("score_breakdown", {}).get("TOTAL SCORE (100)", conf)
                horizon     = sug.get("time_horizon", "Intraday")
                rationale   = sug.get("rationale", "")
                warnings    = sug.get("risk_warnings", [])
                warn_html   = "".join(
                    f"<div style='font-size:9px;color:#ff8888;padding:1px 0;'>[!] {w}</div>"
                    for w in warnings[:2]
                )
                score_pct   = min(100, int(score_total))
                score_clr   = "#00ff41" if score_pct >= 80 else ("#ffd700" if score_pct >= 65 else "#ff3333")
                expiry_html = f"<div style='font-size:9px;color:#00d4ff;margin-bottom:4px;'>{expiry_lbl}</div>" if expiry_lbl else ""

                st.html(f"""<div style="background:#0a0a0a;border:1px solid #1c1c1c;border-top:3px solid {badge_bg};padding:12px 14px;border-radius:4px;font-family:'JetBrains Mono',monospace;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
<span style="background:{badge_bg};color:{badge_txt};font-size:9px;font-weight:700;padding:2px 6px;border-radius:2px;">{action}</span>
<span style="font-size:9px;color:{dir_clr};font-weight:700;">{direction}</span>
</div>
<div style="font-size:13px;font-weight:700;color:#ffffff;margin-bottom:2px;">{inst_name}</div>
{expiry_html}
<div style="font-size:10px;color:#666;margin-bottom:8px;">{horizon}</div>
<div style="margin-bottom:6px;">
<div style="display:flex;justify-content:space-between;font-size:9px;color:#888;margin-bottom:2px;">
<span>Multi-Factor Score</span><span style="color:{score_clr};font-weight:700;">{score_pct}/100</span>
</div>
<div style="background:#1a1a1a;border-radius:2px;height:4px;">
<div style="background:{score_clr};width:{score_pct}%;height:4px;border-radius:2px;"></div>
</div>
</div>
<div style="background:#111;padding:7px;border-radius:3px;font-size:10px;margin-bottom:7px;">
<div style="color:#888;margin-bottom:2px;">Entry Zone</div>
<div style="color:#fff;font-weight:700;margin-bottom:4px;">₹{ez_low:,.2f} – ₹{ez_high:,.2f}</div>
<div style="display:flex;justify-content:space-between;margin-bottom:2px;">
<span style="color:#666;">T1/T2/T3:</span>
<span style="color:#00ff41;font-weight:700;">₹{tgt1:,.0f} / ₹{tgt2:,.0f} / ₹{tgt3:,.0f}</span>
</div>
<div style="display:flex;justify-content:space-between;margin-bottom:2px;">
<span style="color:#666;">Stop Loss:</span>
<span style="color:#ff3333;font-weight:700;">₹{sl:,.1f}</span>
</div>
<div style="display:flex;justify-content:space-between;">
<span style="color:#666;">R:R</span>
<span style="color:#ffd700;font-weight:700;">{rr}</span>
</div>
</div>
<div style="font-size:9px;color:#a0a0a0;line-height:1.35;margin-bottom:5px;">{rationale[:100]}...</div>
{warn_html}
</div>""")

                btn_key = f"term_exec_sug_{idx}_{sug.get('id', idx)}"
                if st.button("Paper Execute", key=btn_key, use_container_width=True):
                    try:
                        import json, os, uuid
                        trade_path = os.path.join("data", "manual_trades.json")
                        os.makedirs("data", exist_ok=True)
                        m_trade = {
                            "id": str(uuid.uuid4())[:8],
                            "ticker": inst_name,
                            "side": "buy" if "BUY" in action else "sell",
                            "qty": 50 if "STOCK" in t_type else 1,
                            "entry": entry_px,
                            "exit": entry_px,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "strategy": f"SUGGESTION_{t_type}",
                        }
                        existing = []
                        if os.path.exists(trade_path):
                            with open(trade_path, "r", encoding="utf-8") as f:
                                existing = json.load(f)
                        existing.append(m_trade)
                        with open(trade_path, "w", encoding="utf-8") as f:
                            json.dump(existing, f)
                        st.success(f"Executed {inst_name} in Paper Portfolio!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Execution error: {e}")

        col_more1, col_more2 = st.columns([3, 1])
        with col_more2:
            if st.button("Explore All Trade Setups & Advisory →", key="btn_goto_trades_bottom", use_container_width=True):
                st.session_state["current_page"] = "Trade Suggestions"
                st.rerun()
    else:
        st.info("Analyzing market flow... Trade recommendations engine is evaluating setups.")

    # ── Upcoming High-Dividend Stocks & Strategic Exit Radar ──────────
    st.markdown('<div class="section-hdr"><h4>Upcoming High-Dividend Stocks & Strategic Exit Radar</h4></div>', unsafe_allow_html=True)
    div_data_quick = data.get("dividends", {})
    up_divs = div_data_quick.get("upcoming", []) if isinstance(div_data_quick, dict) else []
    if not up_divs:
        ca_quick = CorporateActionsPipeline()
        ca_quick._build_upcoming_schedule()
        up_divs = ca_quick.dividend_results.get("upcoming", [])

    if up_divs:
        quick_df = pd.DataFrame(up_divs[:6])
        q_cols = ["Ticker", "Company", "Dividend Amount", "Stock Price", "Yield", "Last Date to Buy", "Ex-Dividend Date", "Record Date", "Hold Until", "Earliest Exit", "When to Exit", "Status"]
        valid_q_cols = [c for c in q_cols if c in quick_df.columns]
        q_styled = quick_df[valid_q_cols].copy()
        
        def _color_q_status(val):
            if isinstance(val, str):
                if "BUY WINDOW" in val: return "color: #00ff41; font-weight: bold"
                if "LAST DAY" in val: return "color: #ff9900; font-weight: bold"
                if "HOLDING" in val: return "color: #00d4ff; font-weight: bold"
            return ""

        st.dataframe(q_styled.style.map(_color_q_status, subset=["Status"] if "Status" in q_styled.columns else []), use_container_width=True, hide_index=True)
        col_div_btn1, col_div_btn2 = st.columns([3, 1])
        with col_div_btn2:
            if st.button("Open Full Dividends Desk & ROI Calculator →", key="btn_goto_divs_bottom", use_container_width=True):
                st.session_state["current_page"] = "Dividends Desk"
                st.rerun()

    # ── Mini Charts Grid & Breadth ───────────────────────────────────────────
    st.markdown('<div class="section-hdr"><h4>Market Breadth & Overview</h4></div>', unsafe_allow_html=True)
    
    # FIX: Loop over all data streams rather than just the initial list
    tickers_with_data = [t for t in data["bars"].keys() if not data["bars"][t].empty]
    
    # Calculate changes and sort by Gainers/Losers
    ticker_stats = []
    advances = 0
    declines = 0
    for tk in tickers_with_data:
        bdf = data["bars"][tk]
        last_c = bdf["close"].iloc[-1]
        prev_close = data["prev_closes"].get(tk, last_c)
        if prev_close == 0: prev_close = last_c
        chg_pct = ((last_c - prev_close) / prev_close * 100) if prev_close else 0
        ticker_stats.append((tk, chg_pct, last_c, bdf))
        if chg_pct >= 0:
            advances += 1
        else:
            declines += 1
            
    # Draw Breadth Bar (Sleek CSS without emojis)
    if tickers_with_data:
        total = len(tickers_with_data)
        adv_pct = (advances / total) * 100
        dec_pct = (declines / total) * 100
        st.markdown(f'''
        <div style="display: flex; height: 4px; border-radius: 0px; overflow: hidden; margin-bottom: 8px;">
            <div style="width: {adv_pct}%; background-color: #00ff41;" title="{advances} Advances"></div>
            <div style="width: {dec_pct}%; background-color: #ff3333;" title="{declines} Declines"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #666666; margin-bottom: 24px; text-transform: uppercase; letter-spacing: 1px;">
            <span>Advances: {advances}</span>
            <span>Declines: {declines}</span>
        </div>
        ''', unsafe_allow_html=True)

    # Sort from highest gainer to lowest loser
    ticker_stats.sort(key=lambda x: x[1], reverse=True)

    if ticker_stats:
        cols_per_row = 5
        for row_start in range(0, min(len(ticker_stats), 15), cols_per_row):
            batch = ticker_stats[row_start:row_start + cols_per_row]
            cols = st.columns(len(batch))
            for col, (tk, chg_pct, last_c, bdf) in zip(cols, batch):
                with col:
                    color = "#00ff41" if chg_pct >= 0 else "#ff3333"
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    
                    # Highlight big moves (>3%) without emojis (use subtle text tags)
                    badge = " [!]" if chg_pct >= 3.0 or chg_pct <= -3.0 else ""
                    
                    st.markdown(f"**{tk}**<span style='color:var(--text-muted); font-size: 10px;'>{badge}</span> <span style='color:{color}'>₹{last_c:.1f} ({chg_pct:+.1f}%)</span>", unsafe_allow_html=True)
                    fig_m = go.Figure(go.Scatter(
                        x=bdf["timestamp"], y=bdf["close"],
                        mode="lines", line=dict(color=color, width=1.5),
                        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.08)",
                    ))
                    fig_m.update_layout(
                        height=80, margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_m, use_container_width=True, key=f"mini_{tk}")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 2: NEWS ROOM
# ═══════════════════════════════════════════════════════════════════════
elif page == "News Room":
    st.markdown("""
    <div class="pro-header">
        <h1>MARKET NEWS & SENTIMENT</h1>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Real-time feeds from CNBC TV18, Moneycontrol, Economic Times, LiveMint, NDTV Profit, Zee Business, Business Standard, Reuters & more")

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        news_filter = st.selectbox("Filter by Category", [
            "All", "Earnings", "IPO", "Regulatory", "Global",
            "Commodities", "Forex", "Sector", "Technical", "Macro",
        ])
    with col_f2:
        ticker_filter = st.selectbox("Filter by Stock", ["All"] + SYSTEM.tickers)
    with col_f3:
        importance_filter = st.selectbox("Importance", ["All", "critical", "high", "normal"])

    # Fetch news
    with st.spinner("Fetching live news from 13 sources..."):
        try:
            news_items = fetch_all_news(max_per_source=10, timeout=6)
            news_df = news_to_dataframe(news_items)
        except Exception:
            news_items = []
            news_df = pd.DataFrame()

    if not news_df.empty:
        # Apply filters
        filtered = news_df.copy()
        if news_filter != "All":
            filtered = filtered[filtered["Category"] == news_filter]
        if ticker_filter != "All":
            filtered = filtered[filtered["Tickers"].str.contains(ticker_filter, na=False)]
        if importance_filter != "All":
            filtered = filtered[filtered["Importance"] == importance_filter]

        st.metric("Total Headlines", f"{len(filtered)} articles from {news_df['Source'].nunique()} sources")

        # Display news cards
        for _, row in filtered.head(50).iterrows():
            imp = row.get("Importance", "normal")
            badge_cls = f"badge-{imp}" if imp in ("critical", "high") else "badge-normal"
            tickers_str = row.get("Tickers", "")
            ticker_badge = f" | {tickers_str}" if tickers_str != "—" else ""

            st.markdown(f"""
            <div class="news-card">
                <span class="{badge_cls}">{imp.upper()}</span>
                <span class="meta" style="margin-left:10px;">{row['Time']} &bull; {row['Source']} &bull; {row['Category']}{ticker_badge}</span>
                <div class="headline" style="margin-top:6px;">{row['Headline']}</div>
                <div class="desc">{row.get('Description', '')[:150]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Could not fetch live news — showing sentiment feed from trading engine instead")
        sent_df = data.get("sentiment_feed", pd.DataFrame())
        if not sent_df.empty:
            def _color_sent(val):
                colors = {"bullish": "#3fb950", "bearish": "#f85149", "neutral": "#388bfd"}
                return f"color: {colors.get(str(val).lower(), 'white')}; font-weight: bold"
            styled = sent_df.style
            if "Sentiment" in sent_df.columns:
                styled = styled.map(_color_sent, subset=["Sentiment"])
            if "Score" in sent_df.columns:
                styled = styled.format({"Score": "{:.2f}"})
            st.dataframe(styled, use_container_width=True, hide_index=True, height=500)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 3: CHART LAB — Professional Interactive Charts
# ═══════════════════════════════════════════════════════════════════════
elif page == "Chart Lab":
    st.markdown("""
    <div class="pro-header">
        <h1>CHART LAB</h1>
    </div>
    """, unsafe_allow_html=True)

    bar_df = data["bars"].get(selected_ticker, pd.DataFrame())

    if bar_df.empty:
        st.warning(f"No price data available for {selected_ticker}. Select a stock from your Trading Portfolio.")
    else:
        # ── Real-Time Live Candle & Tick Transmission Bar ──────────────────
        forming = (data.get("forming_candles") or {}).get(selected_ticker, {})
        last_b = bar_df.iloc[-1]
        live_px = float(forming.get("close") or last_b["close"])
        prev_px = float(bar_df["close"].iloc[-2]) if len(bar_df) > 1 else live_px
        tick_delta = live_px - prev_px
        tick_color = "#00ff41" if tick_delta >= 0 else "#ff3333"
        tick_arrow = "▲" if tick_delta >= 0 else "▼"

        live_vol = float(forming.get("volume") or last_b["volume"])
        buy_vol = float(forming.get("buy_volume") or (live_vol * 0.58 if tick_delta >= 0 else live_vol * 0.42))
        sell_vol = float(forming.get("sell_volume") or (live_vol - buy_vol))
        delta_vol = buy_vol - sell_vol
        delta_color = "#00ff41" if delta_vol >= 0 else "#ff3333"

        now_utc = datetime.now(timezone.utc)
        secs_left = 60 - (now_utc.second % 60)

        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;">
            <div style="flex: 1.2; background: #0a0a0a; border: 1px solid #1c1c1c; border-left: 4px solid {tick_color}; padding: 8px 12px; border-radius: 4px;">
                <div style="font-size: 11px; color: #666666; text-transform: uppercase;">Live Transmission Price</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px;">
                    <span style="font-size: 18px; font-weight: 700; color: {tick_color};">₹{live_px:,.2f} {tick_arrow}</span>
                    <span style="font-size: 11px; color: #a0a0a0;">Tick Δ: <b>{tick_delta:+,.2f}</b></span>
                </div>
            </div>
            <div style="flex: 1; background: #0a0a0a; border: 1px solid #1c1c1c; border-left: 4px solid #00d4ff; padding: 8px 12px; border-radius: 4px;">
                <div style="font-size: 11px; color: #666666; text-transform: uppercase;">Forming Candle (1m)</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px;">
                    <span style="font-size: 14px; font-weight: 600; color: #ffffff;">O: {forming.get('open', last_b['open']):,.1f} H: {forming.get('high', last_b['high']):,.1f} L: {forming.get('low', last_b['low']):,.1f}</span>
                    <span style="font-size: 11px; color: #00d4ff; font-weight: 700;">REFRESH: {secs_left:02d}s</span>
                </div>
            </div>
            <div style="flex: 1.2; background: #0a0a0a; border: 1px solid #1c1c1c; border-left: 4px solid {delta_color}; padding: 8px 12px; border-radius: 4px;">
                <div style="font-size: 11px; color: #666666; text-transform: uppercase;">Order Flow Volume Delta</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px;">
                    <span style="font-size: 14px; font-weight: 700; color: {delta_color};">Δ {delta_vol:+,.0f}</span>
                    <span style="font-size: 11px; color: #a0a0a0;">Buy: <b style="color:#00ff41">{buy_vol:,.0f}</b> │ Sell: <b style="color:#ff3333">{sell_vol:,.0f}</b></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Chart Controls ────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
        with ctrl1:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "OHLC Bars", "Heikin Ashi", "Line", "Area"])
        with ctrl2:
            indicators = st.multiselect("Indicators", [
                "Anchored VWAP (+/-1s, +/-2s Bands)", "Volume Profile (POC/VAH/VAL)",
                "EMA Ribbon (9/21/55/200)", "SMA (20/50/200)", "EMA (9/21)",
                "Bollinger Bands", "VWAP", "Ichimoku Cloud", "Supertrend", "Parabolic SAR",
                "Fibonacci Retracement", "Pivot Points",
            ], default=["Anchored VWAP (+/-1s, +/-2s Bands)", "Volume Profile (POC/VAH/VAL)", "EMA (9/21)"])
        with ctrl3:
            sub_indicators = st.multiselect("Sub-Chart Indicators", [
                "Volume", "Order Flow CVD (Cumulative Delta)", "RSI", "MACD", "Stochastic", "ADX", "OBV",
            ], default=["Volume", "Order Flow CVD (Cumulative Delta)", "RSI"])
        with ctrl4:
            show_trades = st.toggle("Show Trade Markers", value=True)
            show_patterns = st.toggle("Show Candle Patterns", value=False)

        # ── Drawing Tools (in chart config) ───────────────────────────
        st.markdown("""
        <div class="draw-toolbar">
            <b>Drawing Tools:</b> Use Plotly's built-in tools in the chart toolbar (top-right):
            <b>Draw Line</b> │ <b>Draw Rect</b> │ <b>Draw Circle</b> │ <b>Free Draw</b> │ <b>Erase</b>
        </div>
        """, unsafe_allow_html=True)

        # ── Build the chart ───────────────────────────────────────────
        n_subplots = 1 + len(sub_indicators)
        row_heights = [0.6] + [0.4 / max(len(sub_indicators), 1)] * len(sub_indicators) if sub_indicators else [1.0]

        fig = make_subplots(
            rows=n_subplots, cols=1, shared_xaxes=True,
            vertical_spacing=0.02, row_heights=row_heights,
        )

        # ── Main price chart ──────────────────────────────────────────
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=bar_df["timestamp"], open=bar_df["open"], high=bar_df["high"],
                low=bar_df["low"], close=bar_df["close"], name="Price",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            ), row=1, col=1)
        elif chart_type == "OHLC Bars":
            fig.add_trace(go.Ohlc(
                x=bar_df["timestamp"], open=bar_df["open"], high=bar_df["high"],
                low=bar_df["low"], close=bar_df["close"], name="Price",
            ), row=1, col=1)
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=bar_df["close"],
                mode="lines", name="Close", line=dict(color="#58a6ff", width=2),
            ), row=1, col=1)
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=bar_df["close"],
                mode="lines", name="Close", line=dict(color="#58a6ff", width=2),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
            ), row=1, col=1)
        elif chart_type == "Heikin Ashi":
            ha_df = compute_heikin_ashi(bar_df)
            fig.add_trace(go.Candlestick(
                x=ha_df["timestamp"], open=ha_df["open"], high=ha_df["high"],
                low=ha_df["low"], close=ha_df["close"], name="Heikin Ashi",
                increasing_line_color="#00ff41", decreasing_line_color="#ff3333",
            ), row=1, col=1)

        # ── Overlay Indicators ────────────────────────────────────────
        if "Volume Profile (POC/VAH/VAL)" in indicators:
            vp = compute_volume_profile(bar_df)
            if vp["poc"] > 0:
                fig.add_hline(
                    y=vp["poc"], line_color="#ffd700", line_width=2.5, line_dash="solid",
                    annotation_text=f"POC: ₹{vp['poc']:,.2f}",
                    annotation_position="top right",
                    annotation_font=dict(color="#ffd700", size=10, family="JetBrains Mono"),
                    row=1, col=1,
                )
                fig.add_hline(
                    y=vp["vah"], line_color="#00d4ff", line_width=1.2, line_dash="dash",
                    annotation_text=f"VAH: ₹{vp['vah']:,.2f}",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#00d4ff", size=9, family="JetBrains Mono"),
                    row=1, col=1,
                )
                fig.add_hline(
                    y=vp["val"], line_color="#00d4ff", line_width=1.2, line_dash="dash",
                    annotation_text=f"VAL: ₹{vp['val']:,.2f}",
                    annotation_position="top right",
                    annotation_font=dict(color="#00d4ff", size=9, family="JetBrains Mono"),
                    row=1, col=1,
                )

        if "Anchored VWAP (+/-1s, +/-2s Bands)" in indicators and "volume" in bar_df.columns:
            vw, u1, l1, u2, l2 = compute_anchored_vwap_bands(bar_df)
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=vw, mode="lines", name="Anchored VWAP",
                line=dict(color="#ffd700", width=2.5),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=u1, mode="lines", name="VWAP +1σ",
                line=dict(color="#00d4ff", width=1, dash="dot"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=l1, mode="lines", name="VWAP -1σ",
                line=dict(color="#00d4ff", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(0, 212, 255, 0.05)",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=u2, mode="lines", name="VWAP +2σ",
                line=dict(color="#ff3333", width=1.2, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=l2, mode="lines", name="VWAP -2σ",
                line=dict(color="#00ff41", width=1.2, dash="dash"),
            ), row=1, col=1)

        if "EMA Ribbon (9/21/55/200)" in indicators:
            for p, clr, w in [(9, "#00ff41", 1.5), (21, "#00d4ff", 1.5), (55, "#ffbb00", 1.5), (200, "#ff3333", 2.0)]:
                if len(bar_df) >= p:
                    fig.add_trace(go.Scatter(
                        x=bar_df["timestamp"], y=ema(bar_df["close"], p),
                        mode="lines", name=f"EMA {p}",
                        line=dict(color=clr, width=w),
                    ), row=1, col=1)

        if "SMA (20/50/200)" in indicators:
            for period, color in [(20, "#f0883e"), (50, "#a371f7"), (200, "#388bfd")]:
                if len(bar_df) >= period:
                    fig.add_trace(go.Scatter(
                        x=bar_df["timestamp"], y=sma(bar_df["close"], period),
                        mode="lines", name=f"SMA {period}",
                        line=dict(color=color, width=1, dash="dot"),
                    ), row=1, col=1)

        if "EMA (9/21)" in indicators:
            for period, color in [(9, "#3fb950"), (21, "#f85149")]:
                fig.add_trace(go.Scatter(
                    x=bar_df["timestamp"], y=ema(bar_df["close"], period),
                    mode="lines", name=f"EMA {period}",
                    line=dict(color=color, width=1.5),
                ), row=1, col=1)

        if "Bollinger Bands" in indicators:
            bb_u, bb_m, bb_l = bollinger_bands(bar_df["close"])
            fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=bb_u, name="BB Upper",
                                      line=dict(color="#8b949e", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=bb_l, name="BB Lower",
                                      line=dict(color="#8b949e", width=1, dash="dash"),
                                      fill="tonexty", fillcolor="rgba(139,148,158,0.05)"), row=1, col=1)

        if "VWAP" in indicators and "volume" in bar_df.columns:
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=vwap(bar_df),
                mode="lines", name="VWAP",
                line=dict(color="#f778ba", width=2),
            ), row=1, col=1)

        if "Ichimoku Cloud" in indicators and len(bar_df) >= 52:
            tenkan_s, kijun_s, senkou_a, senkou_b_val, chikou = ichimoku(bar_df)
            fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=tenkan_s, name="Tenkan",
                                      line=dict(color="#3fb950", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=kijun_s, name="Kijun",
                                      line=dict(color="#f85149", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=senkou_a, name="Senkou A",
                                      line=dict(color="#3fb950", width=0.5, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=senkou_b_val, name="Senkou B",
                                      line=dict(color="#f85149", width=0.5, dash="dot"),
                                      fill="tonexty", fillcolor="rgba(63,185,80,0.04)"), row=1, col=1)

        if "Supertrend" in indicators and len(bar_df) >= 10:
            st_val, st_dir = supertrend(bar_df)
            colors = ["#3fb950" if d == 1 else "#f85149" for d in st_dir]
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=st_val, name="Supertrend",
                mode="lines+markers",
                marker=dict(size=3, color=colors),
                line=dict(width=2),
            ), row=1, col=1)

        if "Parabolic SAR" in indicators:
            sar_val = parabolic_sar(bar_df)
            fig.add_trace(go.Scatter(
                x=bar_df["timestamp"], y=sar_val, name="SAR",
                mode="markers", marker=dict(size=3, color="#f0883e"),
            ), row=1, col=1)

        if "Fibonacci Retracement" in indicators:
            high_val = bar_df["high"].max()
            low_val = bar_df["low"].min()
            fib = fibonacci_retracement(high_val, low_val)
            for level, price in fib.items():
                fig.add_hline(y=price, line_dash="dash", line_color="#8b949e",
                              opacity=0.4, annotation_text=f"Fib {level} (₹{price:,.1f})",
                              row=1, col=1)

        if "Pivot Points" in indicators:
            pp = pivot_points(bar_df)
            pivot_colors = {"PP": "#58a6ff", "R1": "#f85149", "R2": "#da3633", "R3": "#8b1a1a",
                            "S1": "#3fb950", "S2": "#238636", "S3": "#1a4731"}
            for label, price in pp.items():
                fig.add_hline(y=price, line_dash="dot",
                              line_color=pivot_colors.get(label, "#8b949e"),
                              opacity=0.5, annotation_text=f"{label}: ₹{price:,.1f}",
                              row=1, col=1)

        # ── Trade Markers ─────────────────────────────────────────────
        if show_trades:
            closed_trades = data.get("closed_trades") or []
            ticker_trades = [t for t in closed_trades if t.get("ticker") == selected_ticker]
            if ticker_trades:
                for t in ticker_trades:
                    qty = t.get("qty", 0)
                    entry_p = t.get("entry", 0)
                    exit_p = t.get("exit", 0)
                    pnl_val = t.get("pnl", 0)
                    strat = t.get("strategy", "")
                    closed_at_str = t.get("closed_at", "")

                    if closed_at_str:
                        exit_time = pd.to_datetime(closed_at_str, utc=True)
                        entry_time = exit_time - pd.Timedelta(minutes=15)
                    else:
                        entry_time = bar_df["timestamp"].iloc[0]
                        exit_time = bar_df["timestamp"].iloc[-1]

                    side = t.get("side", "buy").upper()
                    cost = entry_p * qty

                    # Entry marker
                    fig.add_trace(go.Scatter(
                        x=[entry_time], y=[entry_p], mode="markers",
                        name=f"{side} {qty}@₹{entry_p:,.0f}",
                        marker=dict(symbol="triangle-up", size=16, color="#3fb950",
                                    line=dict(width=1.5, color="white")),
                        hovertemplate=(
                            f"<b>{side}</b> {int(qty)} shares<br>"
                            f"@ ₹{entry_p:,.2f} each<br>"
                            f"Total: ₹{cost:,.2f}<br>"
                            f"Strategy: {strat}<extra></extra>"
                        ),
                        showlegend=False,
                    ), row=1, col=1)

                    # Exit marker
                    pnl_color = "#3fb950" if pnl_val >= 0 else "#f85149"
                    fig.add_trace(go.Scatter(
                        x=[exit_time], y=[exit_p], mode="markers",
                        name=f"EXIT ₹{pnl_val:+,.0f}",
                        marker=dict(symbol="triangle-down", size=16, color="#f85149",
                                    line=dict(width=1.5, color="white")),
                        hovertemplate=(
                            f"<b>EXIT</b> {int(qty)} shares<br>"
                            f"@ ₹{exit_p:,.2f}<br>"
                            f"P&L: <b style='color:{pnl_color}'>₹{pnl_val:+,.2f}</b><br>"
                            f"Strategy: {strat}<extra></extra>"
                        ),
                        showlegend=False,
                    ), row=1, col=1)

                    # Connect entry → exit
                    fig.add_trace(go.Scatter(
                        x=[entry_time, exit_time], y=[entry_p, exit_p],
                        mode="lines", line=dict(color=pnl_color, width=1.5, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ), row=1, col=1)

        # ── Candlestick Patterns ──────────────────────────────────────
        if show_patterns and len(bar_df) > 5:
            patterns = detect_patterns(bar_df)
            pattern_names = {"doji": "Doji [Cross]", "hammer": "Hammer",
                             "shooting_star": "Shooting Star",
                             "bullish_engulfing": "Bullish Engulf [UP]",
                             "bearish_engulfing": "Bearish Engulf [DOWN]",
                             "morning_star": "Morning Star",
                             "evening_star": "Evening Star"}
            for pat_col, pat_label in pattern_names.items():
                if pat_col in patterns.columns:
                    mask = patterns[pat_col].fillna(False)
                    if mask.any():
                        bullish = "bullish" in pat_col or pat_col in ("hammer", "morning_star")
                        fig.add_trace(go.Scatter(
                            x=bar_df.loc[mask, "timestamp"],
                            y=bar_df.loc[mask, "high"] * (1.002 if bullish else 0.998),
                            mode="markers+text", text=pat_label,
                            textposition="top center", textfont=dict(size=9),
                            marker=dict(size=8, color="#3fb950" if bullish else "#f85149",
                                        symbol="diamond"),
                            name=pat_label, showlegend=False,
                        ), row=1, col=1)

        # ── Sub-chart indicators ──────────────────────────────────────
        sub_row = 2
        for sub_ind in sub_indicators:
            if sub_ind == "Volume":
                vol_colors = ["#00ff41" if bar_df["close"].iloc[i] >= bar_df["open"].iloc[i]
                              else "#ff3333" for i in range(len(bar_df))]
                fig.add_trace(go.Bar(
                    x=bar_df["timestamp"], y=bar_df["volume"], name="Volume",
                    marker_color=vol_colors, opacity=0.7,
                ), row=sub_row, col=1)
                fig.update_yaxes(title_text="Vol", row=sub_row, col=1)

            elif sub_ind == "Order Flow CVD (Cumulative Delta)":
                bar_delta = np.where(bar_df["close"] >= bar_df["open"], bar_df["volume"] * 0.35, -bar_df["volume"] * 0.35)
                cvd_series = np.cumsum(bar_delta)
                cvd_color = "#00ff41" if cvd_series[-1] >= 0 else "#ff3333"
                fig.add_trace(go.Scatter(
                    x=bar_df["timestamp"], y=cvd_series, mode="lines", name="Order Flow CVD",
                    line=dict(color=cvd_color, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 255, 65, 0.08)" if cvd_series[-1] >= 0 else "rgba(255, 51, 51, 0.08)",
                ), row=sub_row, col=1)
                fig.update_yaxes(title_text="CVD", row=sub_row, col=1)

            elif sub_ind == "RSI":
                rsi_val = rsi(bar_df["close"])
                fig.add_trace(go.Scatter(
                    x=bar_df["timestamp"], y=rsi_val, name="RSI(14)",
                    line=dict(color="#a371f7", width=1.5),
                ), row=sub_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="#f85149", opacity=0.4, row=sub_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", opacity=0.4, row=sub_row, col=1)
                fig.add_hrect(y0=30, y1=70, fillcolor="rgba(56,139,253,0.04)", line_width=0, row=sub_row, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=sub_row, col=1)

            elif sub_ind == "MACD":
                macd_l, sig_l, hist_v = macd(bar_df["close"])
                hist_colors = ["#3fb950" if v >= 0 else "#f85149" for v in hist_v]
                fig.add_trace(go.Bar(x=bar_df["timestamp"], y=hist_v, name="MACD Hist",
                                      marker_color=hist_colors, opacity=0.5), row=sub_row, col=1)
                fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=macd_l, name="MACD",
                                          line=dict(color="#58a6ff", width=1.5)), row=sub_row, col=1)
                fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=sig_l, name="Signal",
                                          line=dict(color="#f0883e", width=1.5)), row=sub_row, col=1)
                fig.update_yaxes(title_text="MACD", row=sub_row, col=1)

            elif sub_ind == "Stochastic":
                k_val, d_val = stochastic(bar_df)
                fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=k_val, name="%K",
                                          line=dict(color="#58a6ff", width=1.5)), row=sub_row, col=1)
                fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=d_val, name="%D",
                                          line=dict(color="#f0883e", width=1.5)), row=sub_row, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="#f85149", opacity=0.4, row=sub_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="#3fb950", opacity=0.4, row=sub_row, col=1)
                fig.update_yaxes(title_text="Stoch", range=[0, 100], row=sub_row, col=1)

            elif sub_ind == "ADX":
                adx_val = adx(bar_df)
                fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=adx_val, name="ADX(14)",
                                          line=dict(color="#f778ba", width=1.5)), row=sub_row, col=1)
                fig.add_hline(y=25, line_dash="dash", line_color="#8b949e", opacity=0.4, row=sub_row, col=1)
                fig.update_yaxes(title_text="ADX", row=sub_row, col=1)

            elif sub_ind == "OBV" and "volume" in bar_df.columns:
                obv_val = obv(bar_df)
                fig.add_trace(go.Scatter(x=bar_df["timestamp"], y=obv_val, name="OBV",
                                          line=dict(color="#388bfd", width=1.5),
                                          fill="tozeroy", fillcolor="rgba(56,139,253,0.06)"),
                              row=sub_row, col=1)
                fig.update_yaxes(title_text="OBV", row=sub_row, col=1)

            sub_row += 1

        # ── Layout with drawing tools ─────────────────────────────────
        chart_height = 500 + len(sub_indicators) * 150
        fig.update_layout(
            height=chart_height,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.5, xanchor="center",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            # Enable drawing tools
            dragmode="drawline",
            newshape=dict(
                line=dict(color="#58a6ff", width=2),
                fillcolor="rgba(88,166,255,0.1)",
            ),
            modebar=dict(
                orientation="v",
                bgcolor="rgba(22,27,34,0.8)",
                activecolor="#58a6ff",
                add=[
                    "drawline", "drawrect", "drawcircle",
                    "drawopenpath", "drawclosedpath", "eraseshape",
                ],
            ),
        )

        config = {
            "displayModeBar": True,
            "modeBarButtonsToAdd": [
                "drawline", "drawrect", "drawcircle",
                "drawopenpath", "eraseshape",
            ],
            "scrollZoom": True,
        }
        st.plotly_chart(fig, use_container_width=True, config=config)

        # ── Quick Stats Panel ─────────────────────────────────────────
        st.markdown('<div class="section-hdr"><h4>Quick Stats</h4></div>', unsafe_allow_html=True)
        qs1, qs2, qs3, qs4, qs5, qs6 = st.columns(6)
        last_close = bar_df["close"].iloc[-1]
        prev_close = bar_df["close"].iloc[-2] if len(bar_df) > 1 else last_close
        day_change = last_close - prev_close
        rsi_now = rsi(bar_df["close"]).iloc[-1] if len(bar_df) > 14 else 50
        atr_now = atr(bar_df).iloc[-1] if len(bar_df) > 14 else 0
        adx_now = adx(bar_df).iloc[-1] if len(bar_df) > 28 else 0
        vol_avg = bar_df["volume"].mean()

        qs1.metric("Last Price", f"₹{last_close:,.2f}", delta=f"₹{day_change:+,.2f}")
        qs2.metric("RSI (14)", f"{rsi_now:.1f}",
                    delta="Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Neutral")
        qs3.metric("ATR (14)", f"₹{atr_now:,.2f}")
        qs4.metric("ADX (14)", f"{adx_now:.1f}",
                    delta="Strong Trend" if adx_now > 25 else "Weak Trend")
        qs5.metric("Avg Volume", f"{vol_avg:,.0f}")
        qs6.metric("Day High/Low", f"₹{bar_df['high'].max():,.1f} / ₹{bar_df['low'].min():,.1f}")



# ═══════════════════════════════════════════════════════════════════════
#  PAGE: OPTIONS DESK — 30-Year Veteran Derivatives Matrix
# ═══════════════════════════════════════════════════════════════════════
elif page == "Options Desk":
    st.markdown("""
    <div class="pro-header">
        <h1>OPTIONS DESK</h1>
    </div>
    """, unsafe_allow_html=True)

    opt_engine = OptionsEngine()

    # Controls Row
    oc1, oc2, oc3 = st.columns([2, 1, 1])
    with oc1:
        selected_underlying = st.selectbox(
            "Underlying Derivative",
            ["NIFTY 50", "BANKNIFTY", "FINNIFTY", "RELIANCE", "HDFCBANK", "TCS", "INFY"],
            index=0,
        )
    with oc2:
        dte_input = st.number_input("Days to Expiry (DTE)", min_value=0.5, max_value=45.0, value=4.0, step=0.5)
    with oc3:
        atm_iv_input = st.number_input(
            "Base ATM IV (%)",
            min_value=5.0, max_value=80.0,
            value=14.5 if "NIFTY" in selected_underlying else 21.0,
            step=0.5,
        )

    # Resolve spot price
    live_chains = data.get("options_chains") or {}
    if selected_underlying in live_chains:
        chain_dict = live_chains[selected_underlying]
        spot_price = float(chain_dict.get("spot_price", 24500.0))
    else:
        benchmarks = {
            "NIFTY 50": 24500.0, "BANKNIFTY": 52100.0, "FINNIFTY": 23400.0,
            "RELIANCE": 2980.0, "HDFCBANK": 1720.0, "TCS": 3950.0, "INFY": 1820.0,
        }
        spot_price = benchmarks.get(selected_underlying, 24500.0)

    # Compute live option chain snapshot
    chain_obj = opt_engine.generate_option_chain(
        underlying=selected_underlying,
        spot_price=spot_price,
        atm_iv=atm_iv_input / 100.0,
        dte_days=dte_input,
        num_strikes=17,
    )

    # Macro Derivatives KPI Pulse Bar
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Spot Price", f"₹{spot_price:,.2f}")
    k2.metric("ATM Strike", f"{chain_obj.atm_strike:,.0f}", delta=f"{spot_price - chain_obj.atm_strike:+,.1f} pts")
    
    pcr_tag = "Oversold (Bullish)" if chain_obj.pcr_oi > 1.25 else ("Overbought (Bearish)" if chain_obj.pcr_oi < 0.75 else "Neutral")
    k3.metric("PCR (Open Interest)", f"{chain_obj.pcr_oi:.2f}", delta=pcr_tag)
    k4.metric("PCR (Volume)", f"{chain_obj.pcr_volume:.2f}")
    
    pain_dist = spot_price - chain_obj.max_pain_strike
    k5.metric("Max Pain Strike", f"{chain_obj.max_pain_strike:,.0f}", delta=f"{pain_dist:+,.0f} pts from spot")
    k6.metric("ATM Volatility", f"{chain_obj.atm_iv:.1f}%", delta=f"IVR: {chain_obj.iv_rank:.0f}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    otab1, otab2, otab3, otab4, otab5 = st.tabs([
        "Option Chain",
        "Strategy Payoff & Breakeven",
        "Volatility Smile & Skew",
        "Max Pain & OI Analysis",
        "Veteran Stress Test",
    ])

    # ── TAB 1: OPTION CHAIN ───────────────────────────────────────────
    with otab1:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Institutional Matrix — CALLS (Left) | STRIKES | PUTS (Right)</p>", unsafe_allow_html=True)
        
        chain_rows = []
        for s in chain_obj.strikes:
            chain_rows.append({
                "Call OI": f"{s.call_oi:,}",
                "Call Chg": f"{s.call_oi_change:+d}",
                "Call Vol": f"{s.call_volume:,}",
                "Call IV%": f"{s.call_iv:.1f}",
                "Delta (C)": f"{s.call_delta:.3f}",
                "Theta (C)": f"{s.call_theta:.2f}",
                "Call LTP": f"₹{s.call_ltp:,.2f}",
                "STRIKE": f"{s.strike:,.0f}",
                "Put LTP": f"₹{s.put_ltp:,.2f}",
                "Delta (P)": f"{s.put_delta:.3f}",
                "Theta (P)": f"{s.put_theta:.2f}",
                "Put IV%": f"{s.put_iv:.1f}",
                "Put Vol": f"{s.put_volume:,}",
                "Put Chg": f"{s.put_oi_change:+d}",
                "Put OI": f"{s.put_oi:,}",
            })

        df_chain = pd.DataFrame(chain_rows)
        st.dataframe(df_chain, use_container_width=True, hide_index=True)
        st.caption(f"ATM Strike highlighted at ₹{chain_obj.atm_strike:,.0f}. Risk-free rate: {chain_obj.risk_free_rate*100:.1f}%. Expiry: {chain_obj.expiry_date} ({chain_obj.dte_days} DTE)")

    # ── TAB 2: STRATEGY PAYOFF & BREAKEVEN ────────────────────────────
    with otab2:
        strat_col1, strat_col2 = st.columns([1, 2])
        with strat_col1:
            strategy_choice = st.selectbox(
                "Options Playbook",
                [
                    "Iron Condor (Delta-Neutral Theta Harvest)",
                    "Iron Butterfly (Pin Strike Decay)",
                    "Short Straddle (Maximum Theta Decay)",
                    "Long Strangle (Volatility Breakout / Event)",
                    "Bull Call Spread (Defined-Risk Upside)",
                    "Bear Put Spread (Defined-Risk Downside)",
                ],
                index=0,
            )
            payoff_res = opt_engine.build_preset_strategy(
                strategy_preset=strategy_choice,
                underlying=selected_underlying,
                spot_price=spot_price,
                dte_days=dte_input,
                atm_iv=atm_iv_input / 100.0,
            )

            st.markdown(f"#### {payoff_res.name}")
            
            # Metrics Grid
            pm1, pm2 = st.columns(2)
            prem_label = "Net Credit" if payoff_res.net_premium >= 0 else "Net Debit"
            pm1.metric(prem_label, f"₹{abs(payoff_res.net_premium):,.2f}")
            pm2.metric("Max Profit", f"₹{payoff_res.max_profit:,.2f}")
            
            pm3, pm4 = st.columns(2)
            pm3.metric("Max Loss", f"₹{payoff_res.max_loss:,.2f}")
            pm4.metric("Risk : Reward", payoff_res.risk_reward_ratio)

            pm5, pm6 = st.columns(2)
            pm5.metric("Probability of Profit (POP)", f"{payoff_res.probability_of_profit:.1f}%")
            pm6.metric("Span Margin (Hedged)", f"₹{payoff_res.margin_required:,.2f}")

            # Breakevens
            be_str = " & ".join([f"₹{b:,.1f}" for b in payoff_res.breakevens]) if payoff_res.breakevens else "None"
            st.markdown(f"**Breakeven Points:** `{be_str}`")

            # Net Portfolio Greeks Card
            st.markdown("""
            <div style="background:#0c0c0c; border:1px solid #222; border-radius:4px; padding:10px; margin-top:8px;">
                <p style="color:#aaa; font-size:0.8rem; margin-bottom:4px; font-family:'JetBrains Mono';">PORTFOLIO GREEKS EXPOSURE</p>
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono'; font-size:0.85rem;">
                    <span>Delta: <b>""" + f"{payoff_res.net_delta:+.2f}" + """</b></span>
                    <span>Gamma: <b>""" + f"{payoff_res.net_gamma:+.4f}" + """</b></span>
                    <span>Theta: <b>""" + f"₹{payoff_res.net_theta:+.2f}/day" + """</b></span>
                    <span>Vega: <b>""" + f"₹{payoff_res.net_vega:+.2f}/1%" + """</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with strat_col2:
            # Interactive Plotly Payoff Chart
            fig_payoff = go.Figure()

            # Zero reference line
            fig_payoff.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)

            # Expiry Payoff Curve
            fig_payoff.add_trace(go.Scatter(
                x=payoff_res.spot_range,
                y=payoff_res.payoff_expiry,
                mode="lines",
                name="Payoff @ Expiry",
                line=dict(color="#00ff41", width=2.5),
            ))

            # T+0 (Today) Payoff Curve
            fig_payoff.add_trace(go.Scatter(
                x=payoff_res.spot_range,
                y=payoff_res.payoff_t0,
                mode="lines",
                name="Payoff @ T+0 (Today)",
                line=dict(color="#00e5ff", width=1.5, dash="dash"),
            ))

            # Current Spot marker
            fig_payoff.add_vline(
                x=spot_price,
                line_dash="dot",
                line_color="#ffffff",
                line_width=1.5,
                annotation_text=f"Spot ₹{spot_price:,.0f}",
                annotation_position="top left",
            )

            # Breakevens markers
            for be in payoff_res.breakevens:
                fig_payoff.add_vline(
                    x=be,
                    line_dash="dash",
                    line_color="#ff9900",
                    line_width=1,
                    annotation_text=f"BE ₹{be:,.0f}",
                    annotation_position="bottom right",
                )

            fig_payoff.update_layout(
                title=f"{selected_underlying} — Payoff Diagram (Expiry vs T+0)",
                title_font=dict(size=14, color="#f0f0f0", family="JetBrains Mono"),
                xaxis_title="Underlying Spot Price (INR)",
                yaxis_title="Profit / Loss (INR)",
                plot_bgcolor="#000000",
                paper_bgcolor="#000000",
                font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40),
                xaxis=dict(showgrid=True, gridcolor="#181818"),
                yaxis=dict(showgrid=True, gridcolor="#181818"),
                height=420,
            )
            st.plotly_chart(fig_payoff, use_container_width=True)

    # ── TAB 3: VOLATILITY SMILE & SKEW ───────────────────────────────
    with otab3:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Implied Volatility (IV) Smile across Strike Ladder showing Crash Put Skew</p>", unsafe_allow_html=True)
        strikes_x = [s.strike for s in chain_obj.strikes]
        call_ivs_y = [s.call_iv for s in chain_obj.strikes]
        put_ivs_y = [s.put_iv for s in chain_obj.strikes]

        fig_skew = go.Figure()
        fig_skew.add_trace(go.Scatter(
            x=strikes_x, y=put_ivs_y,
            mode="lines+markers",
            name="Put IV % (Downside Skew)",
            line=dict(color="#ff3333", width=2),
            marker=dict(size=6),
        ))
        fig_skew.add_trace(go.Scatter(
            x=strikes_x, y=call_ivs_y,
            mode="lines+markers",
            name="Call IV % (Upside Smile)",
            line=dict(color="#00ff41", width=2),
            marker=dict(size=6),
        ))
        fig_skew.add_vline(
            x=spot_price, line_dash="dash", line_color="#ffffff",
            annotation_text=f"Spot ₹{spot_price:,.0f}",
        )
        fig_skew.update_layout(
            title=f"{selected_underlying} — Volatility Smile & Put Skew",
            title_font=dict(size=14, color="#f0f0f0", family="JetBrains Mono"),
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (Annualized %)",
            plot_bgcolor="#000000", paper_bgcolor="#000000",
            font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(showgrid=True, gridcolor="#181818"),
            yaxis=dict(showgrid=True, gridcolor="#181818"),
            height=400,
        )
        st.plotly_chart(fig_skew, use_container_width=True)

        st.markdown("""
        > **30-Year Desk Insight:** Indian index options exhibit a persistent **Volatility Skew** where OTM Puts trade at a 2.5% to 5.0% higher IV than equivalent OTM Calls. This reflects institutional portfolio hedging demand and tail-risk pricing. Veteran traders exploit this skew via **Credit Ratio Spreads** and **Iron Condors** with asymmetric wing widths.
        """)

    # ── TAB 4: MAX PAIN & OI ANALYSIS ────────────────────────────────
    with otab4:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Strike-wise Open Interest (OI) Distribution & Expiry Pinning Point</p>", unsafe_allow_html=True)
        strikes_oi = [s.strike for s in chain_obj.strikes]
        call_ois = [s.call_oi for s in chain_obj.strikes]
        put_ois = [s.put_oi for s in chain_obj.strikes]

        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            x=strikes_oi, y=call_ois,
            name="Call Open Interest (Resistance)",
            marker_color="#ff5252",
        ))
        fig_oi.add_trace(go.Bar(
            x=strikes_oi, y=put_ois,
            name="Put Open Interest (Support)",
            marker_color="#00e676",
        ))
        fig_oi.add_vline(
            x=chain_obj.max_pain_strike,
            line_dash="dot", line_color="#ffd700", line_width=2.5,
            annotation_text=f"MAX PAIN ₹{chain_obj.max_pain_strike:,.0f}",
            annotation_position="top left",
        )
        fig_oi.update_layout(
            barmode="group",
            title=f"{selected_underlying} — Call vs Put Open Interest across Strikes",
            title_font=dict(size=14, color="#f0f0f0", family="JetBrains Mono"),
            xaxis_title="Strike Price",
            yaxis_title="Open Interest (Contracts)",
            plot_bgcolor="#000000", paper_bgcolor="#000000",
            font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(showgrid=True, gridcolor="#181818"),
            yaxis=dict(showgrid=True, gridcolor="#181818"),
            height=400,
        )
        st.plotly_chart(fig_oi, use_container_width=True)

        st.markdown(f"""
        > **Max Pain Theory:** The **Max Pain strike is ₹{chain_obj.max_pain_strike:,.0f}**. On expiry day, option writers (major institutional desks) minimize cumulative cash payouts at this strike. Currently, the spot price is **₹{spot_price:,.2f}**, which is **{pain_dist:+,.1f} points** from Max Pain.
        """)

    # ── TAB 5: VETERAN STRESS TEST ───────────────────────────────────
    with otab5:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Institutional Scenario Matrix — Projected Rupee P&L under simultaneous Spot & Volatility Shocks</p>", unsafe_allow_html=True)
        
        # Test Iron Condor as default scenario portfolio
        lot_s = 50 if "NIFTY" in selected_underlying else 15
        step_s = 50.0 if "NIFTY" in selected_underlying else 100.0
        atm_s = round(spot_price / step_s) * step_s
        sample_legs = [
            OptionLeg(atm_s - 2 * step_s, OptionType.PUT, OptionAction.SELL, 1, 85.0, atm_iv_input / 100.0, lot_s),
            OptionLeg(atm_s - 4 * step_s, OptionType.PUT, OptionAction.BUY, 1, 35.0, atm_iv_input / 100.0, lot_s),
            OptionLeg(atm_s + 2 * step_s, OptionType.CALL, OptionAction.SELL, 1, 75.0, atm_iv_input / 100.0, lot_s),
            OptionLeg(atm_s + 4 * step_s, OptionType.CALL, OptionAction.BUY, 1, 28.0, atm_iv_input / 100.0, lot_s),
        ]
        stress_res = opt_engine.scenario_stress_test(
            legs=sample_legs,
            spot_price=spot_price,
            dte_days=dte_input,
            base_iv=atm_iv_input / 100.0,
        )

        fig_heat = go.Figure(data=go.Heatmap(
            z=stress_res["pnl_matrix"],
            x=stress_res["spot_shocks"],
            y=stress_res["iv_shocks"],
            colorscale=[
                [0.0, "#ff1744"],
                [0.5, "#111111"],
                [1.0, "#00e676"],
            ],
            text=[[f"₹{val:+,.0f}" for val in row] for row in stress_res["pnl_matrix"]],
            texttemplate="%{text}",
            textfont=dict(size=11, family="JetBrains Mono"),
        ))
        fig_heat.update_layout(
            title=f"{selected_underlying} — P&L Stress Heatmap (Spot Shock vs IV Shock)",
            title_font=dict(size=14, color="#f0f0f0", family="JetBrains Mono"),
            xaxis_title="Underlying Spot Price Shift (%)",
            yaxis_title="Implied Volatility Shift (%)",
            plot_bgcolor="#000000", paper_bgcolor="#000000",
            font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
            margin=dict(l=40, r=40, t=50, b=40),
            height=380,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Stress matrix calculated for standard defined-risk Iron Condor. Demonstrates how volatility crush (+Theta) buffers moderate spot moves.")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 4: RISK LAB (Monte Carlo, Correlation & Portfolio Analytics)
# ═══════════════════════════════════════════════════════════════════════
elif page == "Risk Lab":
    st.markdown("""
    <div class="pro-header">
        <h1>RISK LAB & PORTFOLIO ANALYTICS</h1>
        <p>Institutional Stress Testing, Monte Carlo Simulations, and Cross-Asset Correlation</p>
    </div>
    """, unsafe_allow_html=True)

    rtab1, rtab2, rtab3 = st.tabs([
        "Monte Carlo Simulator",
        "Correlation Matrix",
        "Portfolio Risk & Exposure"
    ])

    # ── TAB 1: MONTE CARLO SIMULATOR ─────────────────────────────────
    with rtab1:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Simulate 10,000 stochastic equity paths to quantify Probability of Ruin and Drawdown Risk.</p>", unsafe_allow_html=True)
        
        # User input parameters
        mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
        with mc_c1:
            mc_capital = st.number_input("Starting Equity (₹)", min_value=50000.0, max_value=50000000.0, value=float(snap.get("equity", 1000000.0)), step=50000.0)
            mc_trades = st.slider("Trade Horizon (Trades)", min_value=50, max_value=500, value=252, step=10)
        with mc_c2:
            calc_wr = float(stats.get("win_rate_pct", 55.0)) / 100.0
            calc_wr = max(0.2, min(0.85, calc_wr))
            mc_win_rate = st.slider("Win Rate", min_value=0.20, max_value=0.85, value=round(calc_wr, 2), step=0.01, format="%.2f")
            mc_sims = st.selectbox("Simulations Count", [1000, 2500, 5000, 10000], index=2)
        with mc_c3:
            mc_avg_win = st.slider("Average Win (%)", min_value=0.5, max_value=8.0, value=2.5, step=0.1) / 100.0
        with mc_c4:
            mc_avg_loss = st.slider("Average Loss (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1) / 100.0

        # Run simulation
        mc_results = simulate_equity_paths(
            win_rate=mc_win_rate,
            avg_win=mc_avg_win,
            avg_loss=mc_avg_loss,
            initial_equity=mc_capital,
            n_trades=mc_trades,
            n_sims=mc_sims,
            seed=42,
        )

        # Risk of Ruin Metric Cards
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Median Terminal", f"₹{mc_results['median_terminal']:,.0f}")
        mcol2.metric("Expected CAGR", f"{mc_results['expected_cagr'] * 100.0:+.1f}%")
        mcol3.metric("P(Drawdown ≥ 10%)", f"{mc_results['prob_of_ruin_10pct'] * 100.0:.1f}%")
        mcol4.metric("P(Drawdown ≥ 25%)", f"{mc_results['prob_of_ruin_25pct'] * 100.0:.1f}%")
        ruin_color = "normal" if mc_results['prob_of_ruin_50pct'] < 0.05 else "inverse"
        mcol5.metric("P(Ruin ≥ 50%)", f"{mc_results['prob_of_ruin_50pct'] * 100.0:.1f}%")

        # Percentile Curves for Fan Chart
        paths = mc_results["paths"]
        pct_5 = np.percentile(paths, 5, axis=0)
        pct_25 = np.percentile(paths, 25, axis=0)
        pct_50 = np.percentile(paths, 50, axis=0)
        pct_75 = np.percentile(paths, 75, axis=0)
        pct_95 = np.percentile(paths, 95, axis=0)
        trades_idx = np.arange(1, mc_trades + 1)

        fig_fan = go.Figure()
        # 5th-95th percentile band
        fig_fan.add_trace(go.Scatter(
            x=trades_idx, y=pct_95, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))
        fig_fan.add_trace(go.Scatter(
            x=trades_idx, y=pct_5, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0, 255, 65, 0.08)",
            name="5th - 95th Percentile", hoverinfo="skip"
        ))
        # 25th-75th percentile band
        fig_fan.add_trace(go.Scatter(
            x=trades_idx, y=pct_75, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))
        fig_fan.add_trace(go.Scatter(
            x=trades_idx, y=pct_25, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0, 255, 65, 0.20)",
            name="25th - 75th Percentile", hoverinfo="skip"
        ))
        # Median line
        fig_fan.add_trace(go.Scatter(
            x=trades_idx, y=pct_50, mode="lines",
            line=dict(color="#00ff41", width=2.5),
            name="Median Equity (50th)",
        ))
        # Starting equity benchmark line
        fig_fan.add_hline(y=mc_capital, line_dash="dash", line_color="#666666", annotation_text="Starting Capital", annotation_position="bottom right")

        fig_fan.update_layout(
            title="Monte Carlo Projected Equity Cone (Confidence Bands)",
            title_font=dict(size=14, color="#f0f0f0", family="JetBrains Mono"),
            xaxis_title="Completed Trades",
            yaxis_title="Projected Equity (INR)",
            plot_bgcolor="#000000", paper_bgcolor="#000000",
            font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
            margin=dict(l=50, r=40, t=50, b=40),
            height=420,
            xaxis=dict(showgrid=True, gridcolor="#181818"),
            yaxis=dict(showgrid=True, gridcolor="#181818"),
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        # Terminal Distribution Histogram
        term_eq = mc_results["terminal_equities"]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=term_eq, nbinsx=60,
            marker_color="#00ff41", opacity=0.75,
            name="Terminal Equity"
        ))
        fig_hist.add_vline(x=mc_capital, line_dash="dash", line_color="#ff3333", annotation_text="Break-even", annotation_position="top left")
        fig_hist.update_layout(
            title="Terminal Equity Distribution at Horizon",
            title_font=dict(size=13, color="#f0f0f0", family="JetBrains Mono"),
            xaxis_title="Ending Equity (INR)",
            yaxis_title="Frequency",
            plot_bgcolor="#000000", paper_bgcolor="#000000",
            font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
            margin=dict(l=40, r=40, t=40, b=40),
            height=280,
            xaxis=dict(showgrid=True, gridcolor="#181818"),
            yaxis=dict(showgrid=True, gridcolor="#181818"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── TAB 2: CORRELATION MATRIX ────────────────────────────────────
    with rtab2:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Pairwise return correlation matrix across the active Indian market universe.</p>", unsafe_allow_html=True)
        
        bars_raw = data.get("bars", {})
        # Format bars for correlation function
        bars_dict_fmt = {}
        for tk, bdata in bars_raw.items():
            if isinstance(bdata, pd.DataFrame) and not bdata.empty:
                bars_dict_fmt[tk] = bdata.to_dict("records")
            elif isinstance(bdata, list):
                bars_dict_fmt[tk] = bdata

        if bars_dict_fmt:
            corr_df = compute_correlation_matrix(bars_dict_fmt)
            if not corr_df.empty:
                # Subselect top tickers if universe is large for visual clarity
                disp_tickers = list(corr_df.columns)[:15]
                sub_corr = corr_df.loc[disp_tickers, disp_tickers]

                fig_corr = go.Figure(data=go.Heatmap(
                    z=sub_corr.values,
                    x=sub_corr.columns,
                    y=sub_corr.index,
                    colorscale=[
                        [0.0, "#ff1744"],
                        [0.5, "#111111"],
                        [1.0, "#00ff41"],
                    ],
                    zmin=-1.0, zmax=1.0,
                    text=[[f"{val:.2f}" for val in row] for row in sub_corr.values],
                    texttemplate="%{text}",
                    textfont=dict(size=10, family="JetBrains Mono"),
                ))
                fig_corr.update_layout(
                    title="Cross-Asset 1-Min Return Correlation Matrix",
                    title_font=dict(size=14, color="#f0f0f0", family="JetBrains Mono"),
                    plot_bgcolor="#000000", paper_bgcolor="#000000",
                    font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
                    margin=dict(l=60, r=40, t=50, b=40),
                    height=500,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption("Values range from -1.0 (inverse correlation) to +1.0 (identical movement). Highly correlated assets compound portfolio drawdown risk.")
            else:
                st.info("Insufficient return series data to compute correlation.")
        else:
            st.info("No price bars available for correlation calculation.")

    # ── TAB 3: PORTFOLIO RISK & CONCENTRATION ─────────────────────────
    with rtab3:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Portfolio structure, concentration ratios, and Indian sector exposure analysis.</p>", unsafe_allow_html=True)
        
        pos_raw = data.get("positions", [])
        if isinstance(pos_raw, pd.DataFrame):
            pos_list = pos_raw.to_dict("records")
        else:
            pos_list = pos_raw or []

        port_metrics = compute_portfolio_metrics(bars_dict_fmt, pos_list)

        pr1, pr2, pr3 = st.columns(3)
        hhi = port_metrics.get("herfindahl_index", 0.0)
        hhi_status = "Well Diversified" if hhi < 0.20 else ("Moderate" if hhi < 0.40 else "High Concentration")
        pr1.metric("Herfindahl Index (HHI)", f"{hhi:.4f}", delta=hhi_status, delta_color="off")
        max_c = port_metrics.get("max_concentration_pct", 0.0)
        pr2.metric("Max Single Stock Exposure", f"{max_c:.1f}%")
        p_beta = port_metrics.get("portfolio_beta", 1.0)
        pr3.metric("Portfolio Beta (vs NSE)", f"{p_beta:.2f}")

        # Sector Exposure Donut Chart
        sectors = port_metrics.get("sector_exposure", {})
        if sectors:
            sec_col1, sec_col2 = st.columns([1, 1])
            with sec_col1:
                labels = list(sectors.keys())
                values = list(sectors.values())
                fig_sec = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=0.55,
                    marker=dict(colors=["#00ff41", "#00d4ff", "#ffd700", "#ff3333", "#ff6600", "#9c27b0", "#808080"]),
                    textinfo="label+percent",
                    textfont=dict(family="JetBrains Mono", size=10),
                )])
                fig_sec.update_layout(
                    title="Sector Allocation Breakdown",
                    title_font=dict(size=13, color="#f0f0f0", family="JetBrains Mono"),
                    plot_bgcolor="#000000", paper_bgcolor="#000000",
                    font=dict(color="#a0a0a0", family="JetBrains Mono", size=10),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig_sec, use_container_width=True)

            with sec_col2:
                # Position Size Table
                st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#f0f0f0; margin-top:10px;'>Active Holdings Concentration</p>", unsafe_allow_html=True)
                if pos_list:
                    pos_df_view = pd.DataFrame(pos_list)
                    cols_to_show = [c for c in ["Ticker", "Side", "Qty", "Entry", "Current", "Unrealized PnL"] if c in pos_df_view.columns]
                    st.dataframe(pos_df_view[cols_to_show], use_container_width=True, hide_index=True)
                else:
                    st.caption("No open positions active currently.")
        else:
            st.info("No position data available to calculate sector concentration.")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 5: P&L ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif page == "P&L Analytics":
    st.markdown("""
    <div class="pro-header">
        <h1>P&L ANALYTICS</h1>
    </div>
    """, unsafe_allow_html=True)

    if stats and stats.get("total_trades", 0) > 0:
        # Key metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Trades", stats["total_trades"])
        m2.metric("Win Rate", f"{stats['win_rate_pct']:.1f}%")
        m3.metric("Total P&L", f"₹{stats['total_pnl']:+,.2f}")
        m4.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        m5.metric("Avg Win", f"₹{stats['avg_win']:+,.2f}")
        m6.metric("Avg Loss", f"₹{stats['avg_loss']:+,.2f}")

        m7, m8, m9, m10, m11, m12 = st.columns(6)
        m7.metric("Winners", stats["winning_trades"])
        m8.metric("Losers", stats["losing_trades"])
        m9.metric("Largest Win", f"₹{stats['max_win']:+,.2f}")
        m10.metric("Largest Loss", f"₹{stats['max_loss']:+,.2f}")
        m11.metric("Expectancy", f"₹{stats['expectancy']:+,.2f}")
        m12.metric("Avg Trade", f"₹{stats['avg_trade']:+,.2f}")

        # Charts row
        col_pie, col_dist = st.columns(2)
        with col_pie:
            st.markdown('<div class="section-hdr"><h4>Win / Loss Breakdown</h4></div>', unsafe_allow_html=True)
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Wins", "Losses", "Even"],
                values=[stats["winning_trades"], stats["losing_trades"], stats["even_trades"]],
                marker_colors=["#3fb950", "#f85149", "#388bfd"],
                hole=0.45, textinfo="label+percent+value", textfont_size=13,
            )])
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20),
                                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_dist:
            st.markdown('<div class="section-hdr"><h4>P&L Distribution</h4></div>', unsafe_allow_html=True)
            trades_df = data.get("trades_df", pd.DataFrame())
            if not trades_df.empty and "pnl" in trades_df.columns:
                fig_hist = go.Figure(go.Histogram(
                    x=trades_df["pnl"], nbinsx=20, marker_color="#388bfd",
                ))
                fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20),
                                        xaxis_title="P&L (₹)", yaxis_title="Count",
                                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)

        # Per-trade P&L
        st.markdown('<div class="section-hdr"><h4>Per-Trade P&L</h4></div>', unsafe_allow_html=True)
        if not trades_df.empty and "pnl" in trades_df.columns:
            col_bar, col_cum = st.columns(2)
            with col_bar:
                colors = ["#3fb950" if p >= 0 else "#f85149" for p in trades_df["pnl"]]
                fig_bar = go.Figure(go.Bar(
                    x=trades_df["trade_number"], y=trades_df["pnl"],
                    marker_color=colors, text=trades_df["ticker"], textposition="outside",
                    hovertemplate="Trade #%{x}<br>%{text}<br>₹%{y:+,.2f}<extra></extra>",
                ))
                fig_bar.update_layout(height=350, template="plotly_dark",
                                       xaxis_title="Trade #", yaxis_title="P&L (₹)",
                                       paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_cum:
                fig_cum = go.Figure(go.Scatter(
                    x=trades_df["trade_number"], y=trades_df["cumulative_pnl"],
                    mode="lines+markers", line=dict(color="#3fb950", width=3),
                    fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
                    hovertemplate="Trade #%{x}<br>Cum: ₹%{y:+,.2f}<extra></extra>",
                ))
                fig_cum.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                fig_cum.update_layout(height=350, template="plotly_dark",
                                       xaxis_title="Trade #", yaxis_title="Cumulative P&L (₹)",
                                       paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cum, use_container_width=True)

        # Per-stock performance
        pnl_by_ticker = data.get("pnl_by_ticker", pd.DataFrame())
        if not pnl_by_ticker.empty:
            st.markdown('<div class="section-hdr"><h4>Per-Stock Performance</h4></div>', unsafe_allow_html=True)
            col_tkr, col_wr = st.columns(2)
            with col_tkr:
                sorted_df = pnl_by_ticker.sort_values("total_pnl", ascending=True)
                tkr_colors = ["#3fb950" if p >= 0 else "#f85149" for p in sorted_df["total_pnl"]]
                fig_tkr = go.Figure(go.Bar(
                    y=sorted_df["ticker"], x=sorted_df["total_pnl"], orientation="h",
                    marker_color=tkr_colors,
                    text=[f"₹{v:+,.0f}" for v in sorted_df["total_pnl"]],
                    textposition="outside",
                ))
                fig_tkr.update_layout(height=max(350, len(sorted_df) * 30),
                                       template="plotly_dark", xaxis_title="P&L (₹)",
                                       paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_tkr, use_container_width=True)
            with col_wr:
                sorted_wr = pnl_by_ticker.sort_values("win_rate_pct", ascending=True)
                wr_colors = ["#3fb950" if w >= 60 else "#f85149" if w < 40 else "#388bfd"
                             for w in sorted_wr["win_rate_pct"]]
                fig_wr = go.Figure(go.Bar(
                    y=sorted_wr["ticker"], x=sorted_wr["win_rate_pct"], orientation="h",
                    marker_color=wr_colors,
                    text=[f"{v:.0f}%" for v in sorted_wr["win_rate_pct"]],
                    textposition="outside",
                ))
                fig_wr.add_vline(x=50, line_dash="dash", line_color="white", opacity=0.3)
                fig_wr.update_layout(height=max(350, len(sorted_wr) * 30),
                                      template="plotly_dark", xaxis_title="Win Rate (%)",
                                      xaxis_range=[0, 105], paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_wr, use_container_width=True)
    else:
        st.info("No closed trades yet — analytics will appear after the first trade.")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 5: STRATEGY HUB
# ═══════════════════════════════════════════════════════════════════════
elif page == "Strategy Hub":
    st.markdown("""
    <div class="pro-header">
        <h1>STRATEGY HUB</h1>
    </div>
    """, unsafe_allow_html=True)

    by_strat = data.get("analytics_by_strategy", {})
    if by_strat:
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown('<div class="section-hdr"><h4>Strategy P&L</h4></div>', unsafe_allow_html=True)
            strat_names = list(by_strat.keys())
            strat_pnls = [by_strat[s]["total_pnl"] for s in strat_names]
            strat_colors = ["#3fb950" if p >= 0 else "#f85149" for p in strat_pnls]
            fig_strat = go.Figure(go.Bar(
                x=strat_names, y=strat_pnls, marker_color=strat_colors,
                text=[f"₹{v:+,.0f}" for v in strat_pnls], textposition="outside",
            ))
            fig_strat.update_layout(height=350, template="plotly_dark",
                                     yaxis_title="P&L (₹)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_strat, use_container_width=True)

        with col_s2:
            st.markdown('<div class="section-hdr"><h4>Strategy Win Rates</h4></div>', unsafe_allow_html=True)
            strat_wr = [by_strat[s]["win_rate_pct"] for s in strat_names]
            strat_tc = [by_strat[s]["total_trades"] for s in strat_names]
            fig_wr = go.Figure(go.Bar(
                x=strat_names, y=strat_wr,
                marker_color=["#3fb950" if w >= 50 else "#f85149" for w in strat_wr],
                text=[f"{w:.0f}% ({n}t)" for w, n in zip(strat_wr, strat_tc)],
                textposition="outside",
            ))
            fig_wr.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.3)
            fig_wr.update_layout(height=350, template="plotly_dark",
                                  yaxis_title="Win Rate (%)", yaxis_range=[0, 105],
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_wr, use_container_width=True)

        # Strategy detail table
        st.markdown('<div class="section-hdr"><h4>Strategy Detail</h4></div>', unsafe_allow_html=True)
        strat_rows = []
        for name, s in by_strat.items():
            strat_rows.append({
                "Strategy": name,
                "Trades": s["total_trades"],
                "Win Rate": f"{s['win_rate_pct']:.1f}%",
                "P&L": f"₹{s['total_pnl']:+,.2f}",
                "Profit Factor": f"{s['profit_factor']:.2f}",
                "Avg Win": f"₹{s['avg_win']:+,.2f}",
                "Avg Loss": f"₹{s['avg_loss']:+,.2f}",
                "Expectancy": f"₹{s['expectancy']:+,.2f}",
            })
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

        # Strategy recommendations
        st.markdown('<div class="section-hdr"><h4>AI Strategy Insights</h4></div>', unsafe_allow_html=True)
        best_strat = max(by_strat.items(), key=lambda x: x[1]["total_pnl"])
        worst_strat = min(by_strat.items(), key=lambda x: x[1]["total_pnl"])
        best_wr = max(by_strat.items(), key=lambda x: x[1]["win_rate_pct"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="insight-card insight-green">
                <b style="color:#3fb950; font-size:1.05rem;">BEST P&L: {best_strat[0]}</b><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Total: ₹{best_strat[1]['total_pnl']:+,.2f}</span><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Win Rate: {best_strat[1]['win_rate_pct']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="insight-card insight-blue">
                <b style="color:#58a6ff; font-size:1.05rem;">BEST WIN RATE: {best_wr[0]}</b><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Rate: {best_wr[1]['win_rate_pct']:.1f}%</span><br>
                <span style="color:#8b99a8; font-size:0.9rem;">P&L: ₹{best_wr[1]['total_pnl']:+,.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="insight-card insight-red">
                <b style="color:#f85149; font-size:1.05rem;">NEEDS REVIEW: {worst_strat[0]}</b><br>
                <span style="color:#8b99a8; font-size:0.9rem;">P&L: ₹{worst_strat[1]['total_pnl']:+,.2f}</span><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Win Rate: {worst_strat[1]['win_rate_pct']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Strategy comparison will appear after trades complete")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 6: TRADE LOG
# ═══════════════════════════════════════════════════════════════════════
elif page == "Trade Log":
    st.markdown("""
    <div class="pro-header">
        <h1>EXECUTION LOG</h1>
    </div>
    """, unsafe_allow_html=True)

    trade_df = data["trade_log"]
    if not trade_df.empty:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            log_ticker = st.selectbox("Filter Stock", ["All"] + sorted(trade_df["ticker"].unique().tolist()))
        with fc2:
            log_side = st.selectbox("Filter Side", ["All", "buy", "sell"])
        with fc3:
            log_strat = st.selectbox("Filter Strategy", ["All"] + sorted(trade_df["strategy"].unique().tolist()))

        filtered_trades = trade_df.copy()
        if log_ticker != "All":
            filtered_trades = filtered_trades[filtered_trades["ticker"] == log_ticker]
        if log_side != "All":
            filtered_trades = filtered_trades[filtered_trades["side"] == log_side]
        if log_strat != "All":
            filtered_trades = filtered_trades[filtered_trades["strategy"] == log_strat]

        total_trades = len(filtered_trades)
        total_pnl = filtered_trades["pnl"].sum() if "pnl" in filtered_trades.columns else 0.0
        win_trades = len(filtered_trades[filtered_trades["pnl"] > 0]) if "pnl" in filtered_trades.columns else 0
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Trades Filtered", f"{total_trades}")
        mc2.metric("Total P&L", f"₹{total_pnl:+,.2f}")
        mc3.metric("Win Rate", f"{win_rate:.1f}%")

        display_cols = [c for c in filtered_trades.columns if c not in ("trade_number", "cumulative_pnl", "result")]
        tl = filtered_trades[display_cols].copy() if display_cols else filtered_trades.copy()

        if "pnl" in tl.columns:
            def _color_trade_pnl(val):
                if isinstance(val, (int, float)):
                    return f"color: {'#3fb950' if val >= 0 else '#f85149'}; font-weight: bold"
                return ""
            styled_tl = tl.style.map(_color_trade_pnl, subset=["pnl"]).format({"pnl": "₹{:+,.2f}"})
            st.dataframe(styled_tl, use_container_width=True, hide_index=True, height=600)
        else:
            st.dataframe(tl, use_container_width=True, hide_index=True, height=600)
    else:
        st.info("No trades yet")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: TRADE SUGGESTIONS & INSTITUTIONAL ADVISORY DESK
# ═══════════════════════════════════════════════════════════════════════
elif page in ["Trade Suggestions", "Trade Ideas & Advisory"]:
    st.markdown("""
    <div class="pro-header">
        <h1>INSTITUTIONAL TRADE IDEAS & ADVISORY DESK</h1>
        <p>Quantitative Multi-Factor Strategy Screener: Equities, Directional Options & High-Probability Spreads</p>
    </div>
    """, unsafe_allow_html=True)

    all_sugs = data.get("trade_suggestions") or []
    if not all_sugs:
        recommender = TradeRecommender()
        all_sugs = recommender.generate_recommendations(
            bars_dict=data.get("bars", {}),
            options_chains=data.get("options_chains", {}),
            regime_info=data.get("regime_info", {}),
            prev_closes=data.get("prev_closes", {}),
            account_equity=data.get("snapshot", {}).get("equity", 10_00_000.0),
        )

    # ── Screener & Filter Controls ────────────────────────────────────
    fcol1, fcol2, fcol3 = st.columns([1.5, 1.5, 1])
    with fcol1:
        asset_filter = st.selectbox(
            "Asset Class / Strategy Type",
            ["All Assets", "Index Options (NIFTY/BANKNIFTY)", "Stock Options", "Cash Equities", "Multi-Leg Spreads & Condors"],
            index=0,
            key="sug_asset_filter"
        )
    with fcol2:
        horizon_filter = st.selectbox(
            "Time Horizon",
            ["All Horizons", "Intraday (Exit before 15:15 IST)", "Short Swing (1 - 3 Days)", "Positional (1 - 2 Weeks)"],
            index=0,
            key="sug_horizon_filter"
        )
    with fcol3:
        min_conf = st.slider("Min Conviction %", min_value=70, max_value=95, value=75, step=1, key="sug_min_conf")

    # Filter logic
    filtered_sugs = []
    for s in all_sugs:
        if s.get("confidence_pct", 0) < min_conf:
            continue
        t_type = s.get("trade_type", "")
        if asset_filter == "Index Options (NIFTY/BANKNIFTY)" and ("NIFTY" not in s.get("ticker", "") or "CONDOR" in t_type):
            continue
        elif asset_filter == "Stock Options" and ("STOCK" in t_type or "NIFTY" in s.get("ticker", "")):
            continue
        elif asset_filter == "Cash Equities" and "BUY_STOCK" not in t_type:
            continue
        elif asset_filter == "Multi-Leg Spreads & Condors" and ("CONDOR" not in t_type and "SPREAD" not in t_type):
            continue

        if horizon_filter != "All Horizons" and s.get("time_horizon") != horizon_filter:
            continue
        filtered_sugs.append(s)

    # ── KPI Summary Cards ─────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Actionable Setups", len(filtered_sugs), delta="High Conviction")
    avg_conf = (sum(s.get("confidence_pct", 80) for s in filtered_sugs) / len(filtered_sugs)) if filtered_sugs else 0
    kpi2.metric("Average Conviction", f"{avg_conf:.1f}%", delta="Multi-Factor Confluence")
    top_setup = filtered_sugs[0].get("instrument_name", "N/A") if filtered_sugs else "None"
    kpi3.metric("Top Pick Setup", top_setup)
    kpi4.metric("Capital Utilization Cap", "25.0% Max", delta="Institutional Risk Rule")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if filtered_sugs:
        # Display cards in 2-column responsive layout
        for row_idx in range(0, len(filtered_sugs), 2):
            card_cols = st.columns(2)
            for col_idx in range(2):
                item_idx = row_idx + col_idx
                if item_idx >= len(filtered_sugs):
                    break
                sug = filtered_sugs[item_idx]
                with card_cols[col_idx]:
                    action     = sug.get("action_label", "BUY")
                    t_type     = sug.get("trade_type", "EQUITY")
                    direction  = sug.get("direction", "BULLISH")
                    is_neutral = direction == "NEUTRAL" or "CONDOR" in str(t_type) or "STRANGLE" in str(t_type)
                    is_bull    = (not is_neutral) and (direction == "BULLISH" or ("BUY" in action and "PUT" not in action))
                    is_bear    = (not is_neutral) and (direction == "BEARISH" or "PUT" in action or "SELL" in action)
                    badge_bg   = "#ffd700" if is_neutral else ("#00ff41" if is_bull else "#ff3333")
                    badge_txt  = "#000000" if (is_neutral or is_bull) else "#ffffff"
                    dir_clr    = "#ffd700" if is_neutral else ("#00ff41" if is_bull else "#ff3333")

                    inst_name   = sug.get("instrument_name", sug.get("ticker"))
                    expiry_lbl  = sug.get("expiry_label", "")
                    ticker      = sug.get("ticker", "")
                    spot_px     = sug.get("spot_price", 0.0)
                    entry_px    = sug.get("entry_price", 0.0)
                    ez_low      = sug.get("entry_zone_low",  entry_px * 0.998)
                    ez_high     = sug.get("entry_zone_high", entry_px * 1.002)
                    tgt1        = sug.get("target_1", 0.0)
                    tgt2        = sug.get("target_2", 0.0)
                    tgt3        = sug.get("target_3", 0.0)
                    sl          = sug.get("stop_loss", 0.0)
                    rr          = sug.get("risk_reward", "1 : 2.5")
                    conf        = sug.get("confidence_pct", 80)
                    max_loss    = sug.get("max_loss_pct", 0.0)
                    horizon     = sug.get("time_horizon", "Intraday")
                    rationale   = sug.get("rationale", "")
                    confluences = sug.get("confluence_factors", [])
                    greeks      = sug.get("greeks_profile", {})
                    alloc       = sug.get("capital_allocation_pct", 3.0)
                    warnings    = sug.get("risk_warnings", [])
                    tech_lvls   = sug.get("technical_levels", {})
                    score_bkdn  = sug.get("score_breakdown", {})
                    score_total = score_bkdn.get("TOTAL SCORE (100)", conf)

                    # Build Greek tags
                    greeks_html = ""
                    if greeks:
                        greeks_html = f"""<div style="display:flex;gap:12px;margin-top:8px;font-size:11px;color:#a0a0a0;background:#0e0e0e;padding:7px 10px;border-radius:3px;flex-wrap:wrap;">
<span>Delta: <b style="color:#ffffff;">{greeks.get('delta', 0.0):+.2f}</b></span>
<span>Theta: <b style="color:#ff8888;">{greeks.get('theta', 0.0):.1f}/d</b></span>
<span>Gamma: <b style="color:#00d4ff;">{greeks.get('gamma', 0.0):.4f}</b></span>
<span>Vega: <b style="color:#ffd700;">{greeks.get('vega', 0.0):.1f}</b></span>
<span>IV Rank: <b style="color:#a0ff80;">{greeks.get('iv_rank', greeks.get('vega', 0.0)*0.5+30):.0f}%</b></span>
</div>"""

                    confluence_list_html = "".join([
                        f"<li style='margin-bottom:3px;color:#a0a0a0;'>{cf}</li>" for cf in confluences
                    ])

                    # Technical levels table rows
                    tech_rows_html = ""
                    lvl_map = {
                        "VWAP": "#00d4ff", "EMA_21": "#ffd700", "EMA_50": "#ff9900",
                        "PP": "#888888", "R1": "#00ff41", "S1": "#ff3333",
                        "R2": "#66ff99", "S2": "#ff6666", "RSI_14": "#a0a0a0",
                        "ATR_14": "#a0a0a0",
                    }
                    for k, v in tech_lvls.items():
                        clr = lvl_map.get(k, "#a0a0a0")
                        tech_rows_html += (
                            f"<tr><td style='color:#666;padding:2px 6px;font-size:10px;'>{k}</td>"
                            f"<td style='color:{clr};font-weight:700;padding:2px 6px;font-size:10px;text-align:right;'>"
                            f"{'%' if k=='RSI_14' else '₹'}{v:,.2f}</td></tr>"
                        )

                    # Risk warning badges
                    warn_html = ""
                    for w in warnings:
                        warn_html += f"<div style='background:#1a0505;border-left:3px solid #ff3333;padding:5px 10px;margin-top:4px;font-size:10px;color:#ff8888;border-radius:2px;'>{w}</div>"

                    # Score meter bar
                    score_pct = min(100, int(score_total))
                    score_clr = "#00ff41" if score_pct >= 80 else ("#ffd700" if score_pct >= 65 else "#ff3333")
                    expiry_tag = f"<span style='font-size:11px;color:#00d4ff;margin-left:6px;'>{expiry_lbl}</span>" if expiry_lbl else ""
                    warn_section = f"<div style='margin-top:6px;'><b style='font-size:10px;color:#ff5555;'>Risk Warnings:</b>{warn_html}</div>" if warn_html else ""
                    tech_section = f"<div style='margin-top:10px;'><b style='font-size:10px;color:#666;'>KEY TECHNICAL LEVELS</b><table style='width:100%;margin-top:4px;border-collapse:collapse;'>{tech_rows_html}</table></div>" if tech_rows_html else ""

                    st.html(f"""<div style="background:#0a0a0a;border:1px solid #1c1c1c;border-left:4px solid {badge_bg};padding:16px 18px;border-radius:4px;margin-bottom:4px;font-family:'JetBrains Mono',monospace;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
<div>
<span style="background:{badge_bg};color:{badge_txt};font-size:11px;font-weight:700;padding:3px 8px;border-radius:2px;text-transform:uppercase;">{action}</span>
<span style="background:#1a1a2e;color:{dir_clr};font-size:10px;font-weight:700;padding:2px 7px;border-radius:2px;margin-left:5px;border:1px solid {dir_clr};">{direction}</span>
<span style="font-size:15px;font-weight:700;color:#ffffff;margin-left:8px;">{inst_name}</span>
{expiry_tag}
</div>
<div style="text-align:right;">
<div style="font-size:18px;font-weight:700;color:{score_clr};">{score_pct}<span style="font-size:11px;color:#666;">/100</span></div>
<div style="font-size:9px;color:#666;">SIGNAL SCORE</div>
</div>
</div>
<div style="margin-bottom:10px;">
<div style="background:#1a1a1a;border-radius:3px;height:5px;">
<div style="background:linear-gradient(90deg,{score_clr}80,{score_clr});width:{score_pct}%;height:5px;border-radius:3px;"></div>
</div>
</div>
<div style="font-size:11px;color:#888;margin-bottom:12px;">
Horizon: <b style="color:#fff;">{horizon}</b>
&nbsp;│&nbsp; Alloc: <b style="color:#ffd700;">{alloc}%</b>
&nbsp;│&nbsp; Max Loss: <b style="color:#ff8888;">{max_loss:.1f}%</b>
&nbsp;│&nbsp; Spot: <b>₹{spot_px:,.2f}</b>
</div>
<div style="background:#001a08;border:1px solid #005020;border-radius:4px;padding:8px 12px;margin-bottom:10px;">
<div style="font-size:9px;color:#666;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.5px;">Precision Entry Zone</div>
<div style="font-size:14px;font-weight:700;color:#00ff41;">₹{ez_low:,.2f} &mdash; ₹{ez_high:,.2f}</div>
</div>
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;background:#111;padding:10px;border-radius:4px;margin-bottom:10px;font-size:10px;">
<div style="text-align:center;">
<div style="color:#666;margin-bottom:2px;">ENTRY</div>
<b style="color:#fff;">₹{entry_px:,.1f}</b>
</div>
<div style="text-align:center;">
<div style="color:#666;margin-bottom:2px;">T1</div>
<b style="color:#66ff99;">₹{tgt1:,.1f}</b>
</div>
<div style="text-align:center;">
<div style="color:#666;margin-bottom:2px;">T2</div>
<b style="color:#00ff41;">₹{tgt2:,.1f}</b>
</div>
<div style="text-align:center;">
<div style="color:#666;margin-bottom:2px;">T3 RUNNER</div>
<b style="color:#00d4ff;">₹{tgt3:,.1f}</b>
</div>
<div style="text-align:center;">
<div style="color:#666;margin-bottom:2px;">STOP</div>
<b style="color:#ff3333;">₹{sl:,.1f}</b>
</div>
</div>
<div style="display:inline-block;background:#1a1500;border:1px solid #ffd700;border-radius:3px;padding:3px 10px;font-size:11px;color:#ffd700;font-weight:700;margin-bottom:10px;">
Risk : Reward = {rr}
</div>
{greeks_html}
<div style="margin-top:10px;font-size:12px;color:#e0e0e0;line-height:1.5;">
<b style="color:#00d4ff;">Trade Thesis:</b> {rationale}
</div>
<div style="margin-top:8px;font-size:11px;">
<b style="color:#ffd700;">Quantitative Confluences ({len(confluences)}):</b>
<ul style="margin-top:4px;margin-bottom:6px;padding-left:18px;">
{confluence_list_html}
</ul>
</div>
{warn_section}
{tech_section}
</div>""")

                    # Plotly Score Breakdown Chart
                    if score_bkdn:
                        dims   = [k for k in score_bkdn if k != "TOTAL SCORE (100)"]
                        maxpts = [20, 15, 15, 15, 10, 15, 10]
                        vals   = [score_bkdn.get(d, 0) for d in dims]
                        pcts   = [v / m if m else 0 for v, m in zip(vals, maxpts[:len(dims)])]
                        bar_colors = [
                            "#00ff41" if p >= 0.75 else ("#ffd700" if p >= 0.50 else "#ff3333")
                            for p in pcts
                        ]
                        # Short labels
                        short_labels = [d.split(" (")[0].replace(" Confirmation", "").replace(" Confirm", "") for d in dims]
                        fig_score = go.Figure(go.Bar(
                            x=vals,
                            y=short_labels,
                            orientation="h",
                            marker=dict(color=bar_colors, line=dict(width=0)),
                            text=[f"{v:.1f}/{m}" for v, m in zip(vals, maxpts[:len(dims)])],
                            textposition="outside",
                            textfont=dict(size=9, color="#a0a0a0"),
                        ))
                        fig_score.update_layout(
                            height=180,
                            margin=dict(l=0, r=50, t=4, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(
                                range=[0, 22],
                                gridcolor="#1a1a1a",
                                tickfont=dict(size=8, color="#444"),
                                showticklabels=False,
                            ),
                            yaxis=dict(
                                tickfont=dict(size=9, color="#888", family="JetBrains Mono"),
                                autorange="reversed",
                            ),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})

                    # Interactive Order Execution Box
                    with st.expander(f"Execute Paper Order — {inst_name}", expanded=False):
                        exec_c1, exec_c2, exec_c3 = st.columns([1.1, 1.4, 1.4])
                        with exec_c1:
                            qty_input = st.number_input(
                                "Quantity / Lots",
                                min_value=1,
                                max_value=5000,
                                value=50 if "STOCK" in t_type else 1,
                                step=1 if "STOCK" not in t_type else 10,
                                key=f"sug_qty_{item_idx}_{sug.get('id', item_idx)}"
                            )
                        with exec_c2:
                            outlay = qty_input * entry_px
                            st.html(f"""<div style="background:#0e0e0e;border:1px solid #222;border-radius:3px;padding:8px 12px;margin-top:2px;">
<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Total Capital Outlay</div>
<div style="font-size:15px;font-weight:700;color:#00ff41;font-family:'JetBrains Mono',monospace;">₹{outlay:,.2f}</div>
</div>""")
                        with exec_c3:
                            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                            if st.button("Confirm Paper Execution", key=f"sug_btn_confirm_{item_idx}_{sug.get('id', item_idx)}", use_container_width=True, type="primary"):
                                try:
                                    import json, os, uuid
                                    trade_path = os.path.join("data", "manual_trades.json")
                                    os.makedirs("data", exist_ok=True)
                                    m_trade = {
                                        "id": str(uuid.uuid4())[:8],
                                        "ticker": inst_name,
                                        "side": "buy" if "BUY" in action else "sell",
                                        "qty": qty_input,
                                        "entry": entry_px,
                                        "exit": entry_px,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "strategy": f"SUGGESTION_{t_type}",
                                    }
                                    existing = []
                                    if os.path.exists(trade_path):
                                        with open(trade_path, "r", encoding="utf-8") as f:
                                            existing = json.load(f)
                                    existing.append(m_trade)
                                    with open(trade_path, "w", encoding="utf-8") as f:
                                        json.dump(existing, f)
                                    st.success(f"Executed {qty_input}x {inst_name} at ₹{entry_px:,.2f} in Paper Portfolio!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Execution error: {e}")
    else:
        st.info("No trade suggestions match the current filter criteria. Adjust the filters above to expand results.")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 7: DIVIDENDS & CORPORATE ACTIONS DESK
# ═══════════════════════════════════════════════════════════════════════
elif page in ["Dividends", "Dividends Desk"]:
    st.markdown("""
    <div class="pro-header">
        <h1>DIVIDENDS & CORPORATE ACTIONS DESK</h1>
        <p>Upcoming Dividend Schedule, Cum-Dividend Cutoff Dates, Holding Requirements, and Strategic Exit Windows</p>
    </div>
    """, unsafe_allow_html=True)

    div_data = data.get("dividends", {})
    if isinstance(div_data, dict):
        upcoming_list = div_data.get("upcoming", [])
        hist_list = div_data.get("historical", [])
    elif isinstance(div_data, list):
        upcoming_list = []
        hist_list = div_data
    else:
        upcoming_list = []
        hist_list = []

    # If upcoming_list is empty, build dynamically
    if not upcoming_list:
        ca = CorporateActionsPipeline()
        ca._build_upcoming_schedule()
        upcoming_list = ca.dividend_results.get("upcoming", [])

    dtab1, dtab2, dtab3 = st.tabs([
        "Upcoming Dividends & Trade Execution",
        "Dividend Payout & ROI Calculator",
        "Historical Ex-Date Price Impact"
    ])

    with dtab1:
        # Explanatory Institutional Banner
        st.html("""<div style="background:#0a0a0a;border:1px solid #1c1c1c;border-left:4px solid #00d4ff;padding:14px 18px;border-radius:4px;margin-bottom:16px;font-family:'JetBrains Mono',monospace;">
<div style="font-size:14px;font-weight:700;color:#00d4ff;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">
SEBI T+1 SETTLEMENT DIVIDEND RULES & INSTITUTIONAL DESK PLAYBOOK
</div>
<div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:14px;font-size:11px;color:#a0a0a0;line-height:1.5;">
<div style="background:#111;padding:10px 12px;border-radius:3px;">
<div style="color:#00ff41;font-weight:700;margin-bottom:4px;">1. BUYING & SETTLEMENT RULES (T+1)</div>
• <b>Last Date to Buy (Cum-Date):</b> Must purchase shares on or before this day prior to <b>15:30 IST</b>.<br>
• <b>Ex-Date = Record Date:</b> Under SEBI T+1 rules, trades settle in 1 day. Buying on Cum-Date settles directly into your Demat account on Record Date.<br>
• <b>Buying on Ex-Date:</b> Anyone buying on Ex-Date will NOT receive the dividend.
</div>
<div style="background:#111;padding:10px 12px;border-radius:3px;">
<div style="color:#ffd700;font-weight:700;margin-bottom:4px;">2. MANDATORY HOLDING & EARLIEST EXIT</div>
• <b>Holding Requirement:</b> Must hold shares overnight past the Cum-Date market close (15:30 IST) into Ex-Date open.<br>
• <b>Earliest Exit Time:</b> <b>Ex-Date at 09:15 AM IST (Market Open)</b>.<br>
• <b>Why Selling on Ex-Date Works:</b> A sell order placed on Ex-Date only settles on Ex+1 day. At the end-of-day Record Date cutoff, you remain the registered beneficial owner in CDSL/NSDL records.
</div>
<div style="background:#111;padding:10px 12px;border-radius:3px;">
<div style="color:#ff8888;font-weight:700;margin-bottom:4px;">3. 3 INSTITUTIONAL EXIT STRATEGIES</div>
• <b>Strategy A (Fast Capital Velocity):</b> Exit at 09:15–09:30 AM on Ex-Date. Re-deploy 100% principal immediately into the next dividend stock.<br>
• <b>Strategy B (Pre-Ex Tax Arbitrage):</b> Exit on Cum-Date before 15:30 IST. Captures pre-dividend price run-up, avoiding income slab tax and Section 94(7) loss traps.<br>
• <b>Strategy C (Post-Ex Gap Recovery):</b> Hold 5–15 days post Ex-Date. Blue-chip cash cows typically fill the ex-dividend gap, securing both cash and capital recovery.
</div>
</div>
<div style="background:#150a0a;border:1px solid #331515;border-radius:3px;padding:6px 12px;margin-top:10px;font-size:10px;color:#ff8888;">
<b>TAX & DIVIDEND STRIPPING WARNING (SECTION 94(7)):</b> Dividends in India are taxed at your income slab rate (up to 39%) with 10% TDS under Section 194. If shares are bought within 3 months prior to Record Date and sold within 3 months post Record Date at a loss, capital losses up to the dividend amount cannot be set off against capital gains.
</div>
</div>""")

        if upcoming_list:
            up_df = pd.DataFrame(upcoming_list)

            # Top KPI metrics
            dkpi1, dkpi2, dkpi3, dkpi4 = st.columns(4)
            dkpi1.metric("Upcoming Events", f"{len(up_df)} Stocks")
            max_y = up_df["Yield Num"].max() if "Yield Num" in up_df.columns else 0.0
            max_tk = up_df.loc[up_df["Yield Num"].idxmax(), "Ticker"] if "Yield Num" in up_df.columns else ""
            dkpi2.metric("Highest Yield", f"{max_y:.1f}%", delta=max_tk)
            avg_y = up_df["Yield Num"].mean() if "Yield Num" in up_df.columns else 0.0
            dkpi3.metric("Average Yield", f"{avg_y:.2f}%")
            imminent_cnt = len(up_df[up_df["Days Left to Buy"] <= 7]) if "Days Left to Buy" in up_df.columns else 0
            dkpi4.metric("Buy Window < 7 Days", f"{imminent_cnt} Stocks", delta="Act Soon", delta_color="normal")

            # Filter row
            fcol1, fcol2, fcol3 = st.columns([1.5, 1, 1])
            with fcol1:
                search_tk = st.text_input("Filter by Ticker or Company", value="", placeholder="e.g. VEDL, Coal India, TCS", key="div_search")
            with fcol2:
                min_yield = st.slider("Minimum Yield (%)", min_value=0.0, max_value=8.0, value=1.0, step=0.5, key="div_min_yield")
            with fcol3:
                status_filter = st.selectbox("Status Filter", ["All", "Buy Window Open", "Last Day to Buy Today"], key="div_status_filter")

            filtered_up = up_df.copy()
            if search_tk:
                filtered_up = filtered_up[
                    filtered_up["Ticker"].str.contains(search_tk, case=False, na=False) |
                    filtered_up["Company"].str.contains(search_tk, case=False, na=False)
                ]
            if "Yield Num" in filtered_up.columns:
                filtered_up = filtered_up[filtered_up["Yield Num"] >= min_yield]
            if status_filter == "Buy Window Open" and "Status Code" in filtered_up.columns:
                filtered_up = filtered_up[filtered_up["Status Code"] == "OPEN"]
            elif status_filter == "Last Day to Buy Today" and "Status Code" in filtered_up.columns:
                filtered_up = filtered_up[filtered_up["Status Code"] == "LAST_DAY"]

            # Display table
            display_columns = [
                "Ticker", "Company", "Dividend Amount", "Stock Price", "Yield",
                "Last Date to Buy", "Ex-Dividend Date", "Record Date", "Hold Until",
                "Earliest Exit", "When to Exit", "Status"
            ]
            valid_cols = [c for c in display_columns if c in filtered_up.columns]
            styled_df = filtered_up[valid_cols].copy()

            def _highlight_div_status(val):
                if isinstance(val, str):
                    if "BUY WINDOW OPEN" in val:
                        return "color: #00ff41; font-weight: bold"
                    elif "LAST DAY" in val:
                        return "color: #ff9900; font-weight: bold"
                    elif "HOLDING" in val:
                        return "color: #00d4ff; font-weight: bold"
                    elif "PASSED" in val:
                        return "color: #666666"
                return ""

            styled_table = styled_df.style.map(_highlight_div_status, subset=["Status"] if "Status" in styled_df.columns else [])
            st.dataframe(styled_table, use_container_width=True, hide_index=True, height=450)
        else:
            st.info("No upcoming dividend corporate actions reported at this time.")

    with dtab2:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Simulate expected dividend credit, capital required, and timeline for any upcoming dividend.</p>", unsafe_allow_html=True)
        if upcoming_list:
            calc_tickers = [f"{item['Ticker']} — {item['Company']} ({item['Dividend Amount']}/sh | Yield: {item['Yield']})" for item in upcoming_list]
            sel_item_str = st.selectbox("Select Target Stock", calc_tickers, index=0, key="calc_div_sel")
            sel_tk = sel_item_str.split(" — ")[0]
            target_item = next((item for item in upcoming_list if item["Ticker"] == sel_tk), upcoming_list[0])

            c_col1, c_col2 = st.columns(2)
            with c_col1:
                inv_capital = st.number_input("Deployment Capital (₹)", min_value=10000.0, max_value=50000000.0, value=100000.0, step=10000.0, key="calc_div_cap")
            with c_col2:
                sh_px = target_item.get("Price Num", 2000.0)
                div_per_sh = target_item.get("Amount Num", 15.0)
                calc_qty = int(inv_capital // sh_px)
                st.metric("Shares to Buy", f"{calc_qty:,} shares", delta=f"@ ₹{sh_px:,.2f}/share")

            total_dividend_cash = calc_qty * div_per_sh
            net_inv = calc_qty * sh_px
            payout_pct = (total_dividend_cash / net_inv * 100.0) if net_inv > 0 else 0.0

            res1, res2, res3 = st.columns(3)
            res1.metric("Total Dividend Cash Payout", f"₹{total_dividend_cash:,.2f}", delta="Bank Credit")
            res2.metric("Gross Dividend Return", f"{payout_pct:.2f}%", delta="Single Event")
            res3.metric("Effective Cost Basis Post-Ex", f"₹{max(0, sh_px - div_per_sh):,.2f}")

            # Execution timeline card
            st.html(f"""<div style="background:#0a0a0a;border:1px solid #1c1c1c;padding:14px 18px;border-radius:4px;margin-top:14px;font-family:'JetBrains Mono',monospace;">
<div style="font-size:13px;font-weight:700;color:#f0f0f0;margin-bottom:10px;">
EXECUTION ROADMAP — {target_item['Ticker']} ({target_item['Company']})
</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;font-size:11px;">
<div style="border-left:2px solid #00ff41;padding-left:8px;">
<span style="color:#666;">1. BUY DEADLINE</span><br>
<b style="color:#00ff41;font-size:12px;">{target_item['Last Date to Buy']}</b><br>
<span style="color:#888;">Order must execute before 15:30 IST</span>
</div>
<div style="border-left:2px solid #00d4ff;padding-left:8px;">
<span style="color:#666;">2. HOLD REQUIREMENT</span><br>
<b style="color:#00d4ff;font-size:12px;">Until {target_item['Ex-Dividend Date']} 09:15 AM</b><br>
<span style="color:#888;">Must hold overnight through Cum-Date close</span>
</div>
<div style="border-left:2px solid #ffd700;padding-left:8px;">
<span style="color:#666;">3. EARLIEST EXIT TIME</span><br>
<b style="color:#ffd700;font-size:12px;">{target_item['Ex-Dividend Date']} @ 09:15 AM</b><br>
<span style="color:#888;">Free to sell on Ex-Date; dividend locked</span>
</div>
<div style="border-left:2px solid #ff8888;padding-left:8px;">
<span style="color:#666;">4. DIRECT BANK CREDIT</span><br>
<b style="color:#ff8888;font-size:12px;">Within 30 Days of Ex-Date</b><br>
<span style="color:#888;">Credited via NACH/NEFT (10% TDS if > ₹5k)</span>
</div>
</div>
</div>""")
        else:
            st.info("Upcoming dividend data is loading...")

    with dtab3:
        st.markdown("<p style='font-size:0.85rem; color:#888;'>Ex-Dividend Price Impact: Stock price behavior before and after historical ex-dividend adjustments.</p>", unsafe_allow_html=True)
        if hist_list:
            df_hist = pd.DataFrame(hist_list)
            def _color_impact(val):
                if isinstance(val, str) and "%" in val:
                    num = float(val.replace("%", "").replace("+", ""))
                    return f"color: {'#00ff41' if num >= 0 else '#ff3333'}; font-weight: bold"
                return ""
            styled_hist = df_hist.style.map(_color_impact, subset=["Impact"] if "Impact" in df_hist.columns else [])
            st.dataframe(styled_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No historical impact records found yet. Background scanner is indexing...")


# ═══════════════════════════════════════════════════════════════════════
#  Auto-refresh (1 second)
# ═══════════════════════════════════════════════════════════════════════
# Disable auto-refresh on text-heavy pages so the user can read without interruption
if page in ["Terminal", "Chart Lab", "P&L Analytics"]:
    time.sleep(1)
    _rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if _rerun:
        _rerun()
