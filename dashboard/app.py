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

# ═══════════════════════════════════════════════════════════════════════
#  Page config — dark professional theme
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AlgoTerminal Pro — NSE/BSE/MCX",
    page_icon="🏦",
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
        --bg-primary:    #0a0e14;
        --bg-secondary:  #111820;
        --bg-card:       #131a24;
        --bg-elevated:   #182230;
        --border-subtle: #1e2d3d;
        --border-accent: #253449;
        --text-primary:  #e6edf3;
        --text-secondary:#8b99a8;
        --text-muted:    #5a6876;
        --accent-blue:   #58a6ff;
        --accent-green:  #3fb950;
        --accent-red:    #f85149;
        --accent-orange: #f0883e;
        --accent-purple: #a371f7;
        --accent-pink:   #f778ba;
        --accent-cyan:   #39d2e0;
        --accent-gold:   #e3b341;
        --shadow-sm:     0 1px 3px rgba(0,0,0,0.4);
        --shadow-md:     0 4px 12px rgba(0,0,0,0.5);
        --shadow-lg:     0 8px 30px rgba(0,0,0,0.6);
        --shadow-glow:   0 0 20px rgba(88,166,255,0.15);
        --radius-sm:     6px;
        --radius-md:     10px;
        --radius-lg:     14px;
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
        background: linear-gradient(180deg, #0d1420 0%, #111a26 40%, #0f1720 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: var(--accent-blue);
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
        background: linear-gradient(135deg, #0d151f 0%, #121d2e 40%, #152238 70%, #0d151f 100%);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius-md);
        padding: 16px 24px;
        margin-bottom: 18px;
        box-shadow: var(--shadow-md), inset 0 1px 0 rgba(88,166,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .pro-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green), var(--accent-orange), var(--accent-purple));
    }
    .pro-header h1 {
        background: linear-gradient(90deg, #58a6ff 0%, #3fb950 35%, #e3b341 65%, #f0883e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-size: 1.7rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin: 0;
        text-shadow: none;
    }
    .pro-header p {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin: 4px 0 0;
    }

    /* ── Metric Cards ─────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        box-shadow: var(--shadow-sm);
        transition: all 0.25s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--accent-blue);
        box-shadow: var(--shadow-glow);
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] > div:nth-child(2) > div:first-child {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
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
        border-left: 3px solid var(--accent-blue);
        padding-left: 14px;
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
            equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True)
        else:
            equity_curve = pd.DataFrame(columns=["timestamp", "equity"])

        bars: dict[str, pd.DataFrame] = {}
        for ticker, bar_list in (raw.get("bars") or {}).items():
            df = pd.DataFrame(bar_list)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            bars[ticker] = df

        return {
            "snapshot": snap,
            "positions": positions,
            "bars": bars,
            "equity_curve": equity_curve,
            "sentiment_feed": sentiment_feed,
            "trade_log": trade_log,
            "closed_trades": raw.get("closed_trades") or [],
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
        trade_log["Time"] = pd.to_datetime(trade_log["closed_at"]).dt.strftime("%H:%M")

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
        return live
    return _generate_demo_data()


data = _load_data()
live_mode = is_live()
snap = data["snapshot"]
stats = data.get("analytics") or {}


# ═══════════════════════════════════════════════════════════════════════
#  Sidebar — Professional Controls
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏦 AlgoTerminal Pro")
    st.caption("Institutional-Grade Trading System")
    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["📊 Terminal", "📰 News Room", "📈 Chart Lab", "💰 P&L Analytics",
         "🎯 Strategy Hub", "📋 Trade Log"],
        label_visibility="collapsed",
    )
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

    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)

    if live_mode:
        st.success("🟢 LIVE ENGINE")
    else:
        st.warning("🟡 DEMO MODE")

    if snap.get("halted"):
        st.error("🚨 CIRCUIT BREAKER")
    else:
        st.success("✅ Trading Active")

    st.divider()
    st.caption(f"Drawdown limit: {RISK.max_daily_drawdown_pct:.0%}")
    st.caption(f"Max positions: {RISK.max_open_positions}")
    st.caption(f"Stop-loss: {RISK.default_stop_loss_pct:.0%}")
    st.caption(f"Take-profit: {RISK.default_take_profit_pct:.0%}")
    st.caption(f"Universe size: {len(stock_list)} stocks")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1: TERMINAL (Main Dashboard)
# ═══════════════════════════════════════════════════════════════════════
if page == "📊 Terminal":
    # Header
    st.markdown("""
    <div class="pro-header">
        <h1>AlgoTerminal Pro — NSE / BSE / MCX</h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Ticker Tape ───────────────────────────────────────────────────
    tape_items = []
    for tk in SYSTEM.tickers[:15]:
        bdf = data["bars"].get(tk, pd.DataFrame())
        if not bdf.empty:
            last = bdf["close"].iloc[-1]
            prev = bdf["close"].iloc[-2] if len(bdf) > 1 else last
            chg = last - prev
            chg_pct = (chg / prev * 100) if prev != 0 else 0
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
        eq_df = data["equity_curve"]
        if not eq_df.empty:
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
        st.markdown(f'<div class="section-hdr"><h4>Quick View — {selected_ticker}</h4></div>', unsafe_allow_html=True)
        bar_df = data["bars"].get(selected_ticker, pd.DataFrame())
        if not bar_df.empty:
            fig_q = go.Figure(data=[go.Candlestick(
                x=bar_df["timestamp"],
                open=bar_df["open"], high=bar_df["high"],
                low=bar_df["low"], close=bar_df["close"],
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

    # ── Mini Charts Grid ──────────────────────────────────────────────
    st.markdown('<div class="section-hdr"><h4>Market Overview</h4></div>', unsafe_allow_html=True)
    tickers_with_data = [t for t in SYSTEM.tickers if not data["bars"].get(t, pd.DataFrame()).empty]
    if tickers_with_data:
        cols_per_row = 5
        for row_start in range(0, min(len(tickers_with_data), 15), cols_per_row):
            batch = tickers_with_data[row_start:row_start + cols_per_row]
            cols = st.columns(len(batch))
            for col, tk in zip(cols, batch):
                with col:
                    bdf = data["bars"][tk]
                    last_c = bdf["close"].iloc[-1]
                    first_c = bdf["close"].iloc[0]
                    chg_pct = ((last_c - first_c) / first_c * 100) if first_c else 0
                    color = "#3fb950" if chg_pct >= 0 else "#f85149"
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    st.markdown(f"**{tk}** <span style='color:{color}'>₹{last_c:.1f} ({chg_pct:+.1f}%)</span>", unsafe_allow_html=True)
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
elif page == "📰 News Room":
    st.markdown("""
    <div class="pro-header">
        <h1>📰 Live News Room — Indian Markets</h1>
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
            ticker_badge = f" | 📌 {tickers_str}" if tickers_str != "—" else ""

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
elif page == "📈 Chart Lab":
    st.markdown(f"""
    <div class="pro-header">
        <h1>📈 Chart Lab — {selected_ticker}</h1>
    </div>
    """, unsafe_allow_html=True)

    bar_df = data["bars"].get(selected_ticker, pd.DataFrame())

    if bar_df.empty:
        st.warning(f"No price data available for {selected_ticker}. Select a stock from your Trading Portfolio.")
    else:
        # ── Chart Controls ────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
        with ctrl1:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "OHLC Bars", "Line", "Area", "Heikin Ashi"])
        with ctrl2:
            indicators = st.multiselect("Indicators", [
                "SMA (20/50/200)", "EMA (9/21)", "Bollinger Bands",
                "VWAP", "Ichimoku Cloud", "Supertrend", "Parabolic SAR",
                "Fibonacci Retracement", "Pivot Points",
            ], default=["SMA (20/50/200)"])
        with ctrl3:
            sub_indicators = st.multiselect("Sub-Chart Indicators", [
                "RSI", "MACD", "Stochastic", "ADX", "Volume", "OBV",
            ], default=["Volume", "RSI"])
        with ctrl4:
            show_trades = st.toggle("Show Trade Markers", value=True)
            show_patterns = st.toggle("Show Candle Patterns", value=False)

        # ── Drawing Tools (in chart config) ───────────────────────────
        st.markdown("""
        <div class="draw-toolbar">
            ✏️ <b>Drawing Tools:</b> Use Plotly's built-in tools in the chart toolbar (top-right):
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
            ha_close = (bar_df["open"] + bar_df["high"] + bar_df["low"] + bar_df["close"]) / 4
            ha_open = pd.Series(np.nan, index=bar_df.index)
            ha_open.iloc[0] = (bar_df["open"].iloc[0] + bar_df["close"].iloc[0]) / 2
            for i in range(1, len(bar_df)):
                ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            ha_high = pd.concat([bar_df["high"], ha_open, ha_close], axis=1).max(axis=1)
            ha_low = pd.concat([bar_df["low"], ha_open, ha_close], axis=1).min(axis=1)
            fig.add_trace(go.Candlestick(
                x=bar_df["timestamp"], open=ha_open, high=ha_high,
                low=ha_low, close=ha_close, name="Heikin Ashi",
                increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            ), row=1, col=1)

        # ── Overlay Indicators ────────────────────────────────────────
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
            pattern_names = {"doji": "Doji ⊕", "hammer": "Hammer 🔨",
                             "shooting_star": "Shooting Star ⭐",
                             "bullish_engulfing": "Bullish Engulf ↑",
                             "bearish_engulfing": "Bearish Engulf ↓",
                             "morning_star": "Morning Star 🌅",
                             "evening_star": "Evening Star 🌇"}
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
                vol_colors = ["#3fb950" if bar_df["close"].iloc[i] >= bar_df["open"].iloc[i]
                              else "#f85149" for i in range(len(bar_df))]
                fig.add_trace(go.Bar(
                    x=bar_df["timestamp"], y=bar_df["volume"], name="Volume",
                    marker_color=vol_colors, opacity=0.6,
                ), row=sub_row, col=1)
                fig.update_yaxes(title_text="Vol", row=sub_row, col=1)

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
#  PAGE 4: P&L ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif page == "💰 P&L Analytics":
    st.markdown("""
    <div class="pro-header">
        <h1>💰 Profit & Loss Analytics</h1>
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
elif page == "🎯 Strategy Hub":
    st.markdown("""
    <div class="pro-header">
        <h1>🎯 Strategy Comparison & Analysis</h1>
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
                <b style="color:#3fb950; font-size:1.05rem;">🏆 Best P&L: {best_strat[0]}</b><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Total: ₹{best_strat[1]['total_pnl']:+,.2f}</span><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Win Rate: {best_strat[1]['win_rate_pct']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="insight-card insight-blue">
                <b style="color:#58a6ff; font-size:1.05rem;">📊 Best Win Rate: {best_wr[0]}</b><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Rate: {best_wr[1]['win_rate_pct']:.1f}%</span><br>
                <span style="color:#8b99a8; font-size:0.9rem;">P&L: ₹{best_wr[1]['total_pnl']:+,.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="insight-card insight-red">
                <b style="color:#f85149; font-size:1.05rem;">⚠️ Needs Review: {worst_strat[0]}</b><br>
                <span style="color:#8b99a8; font-size:0.9rem;">P&L: ₹{worst_strat[1]['total_pnl']:+,.2f}</span><br>
                <span style="color:#8b99a8; font-size:0.9rem;">Win Rate: {worst_strat[1]['win_rate_pct']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Strategy comparison will appear after trades complete")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 6: TRADE LOG
# ═══════════════════════════════════════════════════════════════════════
elif page == "📋 Trade Log":
    st.markdown("""
    <div class="pro-header">
        <h1>📋 Complete Trade Log</h1>
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

        st.metric("Showing", f"{len(filtered_trades)} trades")

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
#  Auto-refresh
# ═══════════════════════════════════════════════════════════════════════
if auto_refresh:
    time.sleep(5)
    st.rerun()
