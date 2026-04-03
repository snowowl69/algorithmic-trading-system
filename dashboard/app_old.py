"""
Live Monitoring Dashboard — Indian Markets (NSE / BSE / MCX).

Streamlit application showing all trades, stocks, and analytics in INR (₹).
Reads live trading state from the shared JSON state store written by the
trading engine.  Falls back to realistic demo data when the engine is not
running.

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

# ═══════════════════════════════════════════════════════════════════════
#  Page config
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Algo Trading — NSE/BSE/MCX Live Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Currency helper
CUR = SYSTEM.currency_symbol  # ₹


# ═══════════════════════════════════════════════════════════════════════
#  Data loading — live state store OR demo fallback
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
            "closed_trades": raw.get("closed_trades") or raw.get("trade_log") or [],
        }
    except (KeyError, TypeError):
        return None


def _generate_demo_data() -> dict:
    """Generate realistic demo data for Indian markets when the live engine is not running."""
    np.random.seed(42)
    now = datetime.now(timezone.utc)
    tickers = SYSTEM.tickers

    # Import INR base prices from historical_loader
    from data_engine.historical_loader import _INR_BASE_PRICES

    # OHLCV bars — one per ticker
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
        {"Ticker": "HDFCBANK", "Side": "BUY", "Qty": 40, "Entry": 1638.50,
         "Current": 1662.30, "Unrealized PnL": 952.00, "Stop Loss": 1605.73,
         "Take Profit": 1704.04, "Strategy": "sentiment_extreme"},
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

    headlines = [
        ("RELIANCE", "Reliance Jio 5G rollout accelerates in tier-2 cities", "ET", "bullish", 0.85),
        ("TCS", "TCS bags $2B deal from major European bank", "Moneycontrol", "bullish", 0.88),
        ("INFY", "Infosys revises guidance downward amid macro headwinds", "NDTV Profit", "bearish", -0.62),
        ("HDFCBANK", "HDFC Bank NPA ratio hits record low", "LiveMint", "bullish", 0.79),
        ("ICICIBANK", "ICICI Bank Q3 profit jumps 35% YoY", "Business Standard", "bullish", 0.82),
        ("SBIN", "SBI pension fund AUM crosses ₹5 lakh crore", "ET", "bullish", 0.71),
        ("TATAMOTORS", "Tata Motors JLR margins under pressure", "Bloomberg Quint", "bearish", -0.55),
        ("ITC", "ITC hotel demerger gains SEBI approval", "Moneycontrol", "bullish", 0.76),
        ("GOLD", "Gold hits fresh all-time high on global uncertainty", "Reuters India", "bullish", 0.91),
        ("SILVER", "Silver demand surges on EV battery shift", "ET", "bullish", 0.74),
        ("BAJFINANCE", "Bajaj Finance AUM growth beats estimates", "NDTV Profit", "bullish", 0.80),
        ("ADANIENT", "Adani Enterprises wins new airport contract", "LiveMint", "bullish", 0.68),
    ]
    sentiment_feed = pd.DataFrame([{
        "Time": (now - timedelta(minutes=len(headlines) - i)).strftime("%H:%M"),
        "Ticker": tk, "Headline": hl, "Source": src,
        "Sentiment": sent, "Score": sc,
    } for i, (tk, hl, src, sent, sc) in enumerate(headlines)])

    # Demo trade log with realistic P&L in INR
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
        {"ticker": "WIPRO", "side": "buy", "entry": 555.00, "exit": 562.30, "qty": 120, "pnl": 876.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T07:00:00"},
        {"ticker": "LT", "side": "buy", "entry": 3580.00, "exit": 3620.00, "qty": 15, "pnl": 600.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T07:15:00"},
        {"ticker": "MARUTI", "side": "buy", "entry": 12400.00, "exit": 12280.00, "qty": 5, "pnl": -600.00, "strategy": "sentiment_extreme", "closed_at": "2026-04-01T07:30:00"},
        {"ticker": "GOLD", "side": "buy", "entry": 72000.00, "exit": 72680.00, "qty": 5, "pnl": 3400.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T07:45:00"},
        {"ticker": "SILVER", "side": "buy", "entry": 87500.00, "exit": 86800.00, "qty": 3, "pnl": -2100.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T08:00:00"},
        {"ticker": "CRUDEOIL", "side": "buy", "entry": 6750.00, "exit": 6820.00, "qty": 20, "pnl": 1400.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T08:15:00"},
        {"ticker": "BAJFINANCE", "side": "buy", "entry": 7350.00, "exit": 7420.00, "qty": 10, "pnl": 700.00, "strategy": "sentiment_extreme", "closed_at": "2026-04-01T08:30:00"},
        {"ticker": "ADANIENT", "side": "buy", "entry": 3180.00, "exit": 3150.00, "qty": 20, "pnl": -600.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T08:45:00"},
        {"ticker": "SUNPHARMA", "side": "buy", "entry": 1730.00, "exit": 1762.00, "qty": 30, "pnl": 960.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T09:00:00"},
        {"ticker": "TITAN", "side": "buy", "entry": 3320.00, "exit": 3290.00, "qty": 15, "pnl": -450.00, "strategy": "sentiment_extreme", "closed_at": "2026-04-01T09:15:00"},
        {"ticker": "KOTAKBANK", "side": "buy", "entry": 1880.00, "exit": 1912.00, "qty": 25, "pnl": 800.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T09:30:00"},
        {"ticker": "NESTLEIND", "side": "buy", "entry": 2330.00, "exit": 2365.00, "qty": 20, "pnl": 700.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T09:45:00"},
        {"ticker": "RELIANCE", "side": "buy", "entry": 2970.00, "exit": 2945.00, "qty": 30, "pnl": -750.00, "strategy": "mean_reversion_zscore", "closed_at": "2026-04-01T10:00:00"},
        {"ticker": "NATURALGAS", "side": "buy", "entry": 245.00, "exit": 252.00, "qty": 100, "pnl": 700.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T10:15:00"},
        {"ticker": "COPPER", "side": "buy", "entry": 815.00, "exit": 808.00, "qty": 50, "pnl": -350.00, "strategy": "sentiment_extreme", "closed_at": "2026-04-01T10:30:00"},
        {"ticker": "AXISBANK", "side": "buy", "entry": 1170.00, "exit": 1192.00, "qty": 40, "pnl": 880.00, "strategy": "momentum_ma_crossover", "closed_at": "2026-04-01T10:45:00"},
    ]

    analytics = TradeAnalytics(trades=demo_trades)

    snapshot = {
        "equity": float(eq[-1]),
        "cash": float(eq[-1] * 0.55),
        "daily_pnl": float(eq[-1] - 10_00_000),
        "daily_pnl_pct": float((eq[-1] - 10_00_000) / 10_00_000 * 100),
        "drawdown_pct": 0.8,
        "peak_equity": float(max(eq)),
        "open_positions": 4,
        "halted": False,
    }

    trade_log = pd.DataFrame(demo_trades)
    if "closed_at" in trade_log.columns:
        trade_log["Time"] = pd.to_datetime(trade_log["closed_at"]).dt.strftime("%H:%M")

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
        # Compute analytics from closed trades
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
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙ Controls")
    auto_refresh = st.toggle("Auto-refresh (5 s)", value=True)
    selected_ticker = st.selectbox("Chart Ticker", SYSTEM.tickers)
    st.divider()

    if live_mode:
        st.success("🟢 LIVE — Connected to trading engine")
    else:
        st.warning("🟡 DEMO — Showing simulated data")

    st.divider()
    st.caption(f"Max drawdown limit: {RISK.max_daily_drawdown_pct:.0%}")
    st.caption(f"Max open positions: {RISK.max_open_positions}")
    st.caption(f"Stop-loss default: {RISK.default_stop_loss_pct:.0%}")
    st.caption(f"Take-profit default: {RISK.default_take_profit_pct:.0%}")
    st.caption(f"Tickers tracked: {len(SYSTEM.tickers)}")

    if snap.get("halted", False):
        st.error("🚨 CIRCUIT BREAKER ACTIVE — Trading halted")
    else:
        st.success("✅ Trading active")


# ═══════════════════════════════════════════════════════════════════════
#  Row 1: Key Metrics
# ═══════════════════════════════════════════════════════════════════════
st.markdown("## 📊 NSE / BSE / MCX — Live Trading Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity", f"₹{snap['equity']:,.2f}")
c2.metric(
    "Daily PnL",
    f"₹{snap['daily_pnl']:,.2f}",
    delta=f"{snap['daily_pnl_pct']:.2f}%",
)
c3.metric("Cash", f"₹{snap['cash']:,.2f}")
c4.metric("Drawdown", f"{snap.get('drawdown_pct', 0):.2f}%")
c5.metric("Open Positions", snap.get("open_positions", 0))


# ═══════════════════════════════════════════════════════════════════════
#  Row 2: Profit / Loss Analytics
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 💰 Profit / Loss Analytics")

if stats and stats.get("total_trades", 0) > 0:
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

    # ── Win / Loss Breakdown: Pie + Distribution ──
    col_pie, col_dist = st.columns(2)

    with col_pie:
        st.subheader("Win / Loss Breakdown")
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Wins", "Losses", "Even"],
            values=[stats["winning_trades"], stats["losing_trades"], stats["even_trades"]],
            marker_colors=["#00cc96", "#ef553b", "#636efa"],
            hole=0.45,
            textinfo="label+percent+value",
            textfont_size=14,
        )])
        fig_pie.update_layout(
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_dark", showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_dist:
        st.subheader("P&L Distribution")
        trades_df = data.get("trades_df", pd.DataFrame())
        if not trades_df.empty and "pnl" in trades_df.columns:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=trades_df["pnl"],
                nbinsx=20,
                name="Trade P&L",
                marker_color="#636efa",
            ))
            fig_hist.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="P&L (₹)", yaxis_title="Count",
                template="plotly_dark",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No trade data for distribution")
else:
    st.info("No closed trades yet — analytics will appear after the first trade.")


# ═══════════════════════════════════════════════════════════════════════
#  Row 3: Per-Trade Charts
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📈 Every Trade — Profit & Loss")

trades_df = data.get("trades_df", pd.DataFrame())

if not trades_df.empty and "pnl" in trades_df.columns:
    col_bar, col_cum = st.columns(2)

    with col_bar:
        st.subheader("Per-Trade P&L")
        colors = ["#00cc96" if p >= 0 else "#ef553b" for p in trades_df["pnl"]]
        fig_bar = go.Figure(data=[go.Bar(
            x=trades_df["trade_number"],
            y=trades_df["pnl"],
            marker_color=colors,
            text=trades_df["ticker"],
            textposition="outside",
            hovertemplate=(
                "Trade #%{x}<br>Ticker: %{text}<br>P&L: ₹%{y:+,.2f}<extra></extra>"
            ),
        )])
        fig_bar.update_layout(
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Trade #", yaxis_title="P&L (₹)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_cum:
        st.subheader("Cumulative P&L")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=trades_df["trade_number"],
            y=trades_df["cumulative_pnl"],
            mode="lines+markers",
            name="Cumulative P&L",
            line=dict(color="#00cc96", width=3),
            fill="tozeroy",
            fillcolor="rgba(0,204,150,0.15)",
            hovertemplate="Trade #%{x}<br>Cumulative: ₹%{y:+,.2f}<extra></extra>",
        ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_cum.update_layout(
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Trade #", yaxis_title="Cumulative P&L (₹)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # Per-trade percentage returns
    if "pnl_pct" in trades_df.columns:
        st.subheader("Trade Returns (%)")
        pct_colors = ["#00cc96" if p >= 0 else "#ef553b" for p in trades_df["pnl_pct"]]
        fig_pct = go.Figure(data=[go.Bar(
            x=trades_df["trade_number"],
            y=trades_df["pnl_pct"],
            marker_color=pct_colors,
            text=trades_df["ticker"],
            textposition="outside",
            hovertemplate="Trade #%{x}<br>Ticker: %{text}<br>Return: %{y:+.2f}%<extra></extra>",
        )])
        fig_pct.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Trade #", yaxis_title="Return (%)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_pct, use_container_width=True)
else:
    st.info("Trades will appear here after the first completed trade")


# ═══════════════════════════════════════════════════════════════════════
#  Row 4: Per-Stock Performance
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🏢 Per-Stock Performance")

pnl_by_ticker = data.get("pnl_by_ticker", pd.DataFrame())

if not pnl_by_ticker.empty:
    col_tkr_bar, col_tkr_wr = st.columns(2)

    with col_tkr_bar:
        st.subheader("P&L by Stock")
        sorted_df = pnl_by_ticker.sort_values("total_pnl", ascending=True)
        tkr_colors = ["#00cc96" if p >= 0 else "#ef553b" for p in sorted_df["total_pnl"]]
        fig_tkr = go.Figure(data=[go.Bar(
            y=sorted_df["ticker"],
            x=sorted_df["total_pnl"],
            orientation="h",
            marker_color=tkr_colors,
            text=[f"₹{v:+,.0f}" for v in sorted_df["total_pnl"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>P&L: ₹%{x:+,.2f}<extra></extra>",
        )])
        fig_tkr.update_layout(
            height=max(400, len(sorted_df) * 30),
            margin=dict(l=20, r=80, t=30, b=20),
            xaxis_title="Total P&L (₹)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_tkr, use_container_width=True)

    with col_tkr_wr:
        st.subheader("Win Rate by Stock")
        sorted_wr = pnl_by_ticker.sort_values("win_rate_pct", ascending=True)
        wr_colors = [
            "#00cc96" if wr >= 60 else "#ef553b" if wr < 40 else "#636efa"
            for wr in sorted_wr["win_rate_pct"]
        ]
        fig_wr = go.Figure(data=[go.Bar(
            y=sorted_wr["ticker"],
            x=sorted_wr["win_rate_pct"],
            orientation="h",
            marker_color=wr_colors,
            text=[f"{v:.0f}%" for v in sorted_wr["win_rate_pct"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>",
        )])
        fig_wr.add_vline(x=50, line_dash="dash", line_color="white", opacity=0.4)
        fig_wr.update_layout(
            height=max(400, len(sorted_wr) * 30),
            margin=dict(l=20, r=80, t=30, b=20),
            xaxis_title="Win Rate (%)",
            xaxis_range=[0, 105],
            template="plotly_dark",
        )
        st.plotly_chart(fig_wr, use_container_width=True)

    # Stock detail table
    st.subheader("Stock Detail Table")
    by_tkr = data.get("analytics_by_ticker", {})
    if by_tkr:
        rows = []
        for tkr, s in by_tkr.items():
            rows.append({
                "Ticker": tkr,
                "Trades": s["total_trades"],
                "Wins": s["winning_trades"],
                "Losses": s["losing_trades"],
                "Win Rate": f"{s['win_rate_pct']:.1f}%",
                "Total P&L": f"₹{s['total_pnl']:+,.2f}",
                "Avg Win": f"₹{s['avg_win']:+,.2f}",
                "Avg Loss": f"₹{s['avg_loss']:+,.2f}",
                "Best": f"₹{s['max_win']:+,.2f}",
                "Worst": f"₹{s['max_loss']:+,.2f}",
            })
        detail_df = pd.DataFrame(rows)

        def _color_pnl_str(val):
            if isinstance(val, str) and val.startswith("₹"):
                cleaned = val.replace("₹", "").replace(",", "").replace("+", "")
                try:
                    num = float(cleaned)
                    return f"color: {'#00cc96' if num >= 0 else '#ef553b'}; font-weight: bold"
                except ValueError:
                    return ""
            return ""

        styled = detail_df.style.map(_color_pnl_str, subset=["Total P&L"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("Per-stock analytics will appear after trades complete")


# ═══════════════════════════════════════════════════════════════════════
#  Row 5: Equity Curve + Candlestick
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📉 Charts")
col_eq, col_candle = st.columns([1, 2])

with col_eq:
    st.subheader("Equity Curve")
    eq_df = data["equity_curve"]
    if not eq_df.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df["timestamp"], y=eq_df["equity"],
            mode="lines", name="Equity",
            line=dict(color="#00cc96", width=2),
            fill="tozeroy", fillcolor="rgba(0,204,150,0.1)",
        ))
        fig_eq.update_layout(
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="INR", xaxis_title="",
            template="plotly_dark",
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Waiting for equity data…")

with col_candle:
    st.subheader(f"Candlestick — {selected_ticker}")
    bar_df = data["bars"].get(selected_ticker, pd.DataFrame())
    if not bar_df.empty:
        fig_candle = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
        fig_candle.add_trace(
            go.Candlestick(
                x=bar_df["timestamp"],
                open=bar_df["open"], high=bar_df["high"],
                low=bar_df["low"], close=bar_df["close"],
                name="OHLC",
            ),
            row=1, col=1,
        )
        fig_candle.add_trace(
            go.Bar(
                x=bar_df["timestamp"], y=bar_df["volume"],
                name="Volume", marker_color="rgba(100,149,237,0.5)",
            ),
            row=2, col=1,
        )

        # ── Overlay BUY / SELL trade markers on the candlestick chart ──
        closed_trades = data.get("closed_trades") or []
        ticker_trades = [t for t in closed_trades if t.get("ticker") == selected_ticker]
        if ticker_trades:
            buy_times, buy_prices, buy_texts = [], [], []
            sell_times, sell_prices, sell_texts = [], [], []
            for t in ticker_trades:
                qty = t.get("qty", 0)
                entry_p = t.get("entry", t.get("entry_price", 0))
                exit_p = t.get("exit", t.get("exit_price", 0))
                pnl = t.get("pnl", 0)
                strategy = t.get("strategy", "")
                closed_at_str = t.get("closed_at", "")

                # Entry (buy/sell open)
                if closed_at_str:
                    trade_time = pd.to_datetime(closed_at_str, utc=True)
                    # Place entry a few bars before close
                    entry_time = trade_time - pd.Timedelta(minutes=15)
                else:
                    entry_time = bar_df["timestamp"].iloc[0]
                    trade_time = bar_df["timestamp"].iloc[-1]

                side = t.get("side", "buy").upper()
                cost = entry_p * qty

                buy_times.append(entry_time)
                buy_prices.append(entry_p)
                buy_texts.append(
                    f"<b>{side}</b> {int(qty)} shares<br>"
                    f"@ ₹{entry_p:,.2f} each<br>"
                    f"Total: ₹{cost:,.2f}<br>"
                    f"Strategy: {strategy}"
                )

                sell_times.append(trade_time)
                sell_prices.append(exit_p)
                pnl_color = "#00cc96" if pnl >= 0 else "#ef553b"
                sell_texts.append(
                    f"<b>EXIT</b> {int(qty)} shares<br>"
                    f"@ ₹{exit_p:,.2f} each<br>"
                    f"P&L: <span style='color:{pnl_color}'>₹{pnl:+,.2f}</span><br>"
                    f"Strategy: {strategy}"
                )

            # Entry markers (green triangles for BUY, red for SELL)
            fig_candle.add_trace(
                go.Scatter(
                    x=buy_times, y=buy_prices,
                    mode="markers",
                    name="Entry",
                    marker=dict(
                        symbol="triangle-up", size=14, color="#00cc96",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    text=buy_texts,
                ),
                row=1, col=1,
            )

            # Exit markers (red triangles down)
            fig_candle.add_trace(
                go.Scatter(
                    x=sell_times, y=sell_prices,
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="triangle-down", size=14, color="#ef553b",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    text=sell_texts,
                ),
                row=1, col=1,
            )

            # Draw lines connecting entry → exit for each trade
            for i in range(len(buy_times)):
                pnl_val = ticker_trades[i].get("pnl", 0)
                line_color = "#00cc96" if pnl_val >= 0 else "#ef553b"
                fig_candle.add_trace(
                    go.Scatter(
                        x=[buy_times[i], sell_times[i]],
                        y=[buy_prices[i], sell_prices[i]],
                        mode="lines",
                        line=dict(color=line_color, width=1.5, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=1,
                )

        fig_candle.update_layout(
            height=480, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_candle, use_container_width=True)
    else:
        st.info(f"No bar data for {selected_ticker}")


# ═══════════════════════════════════════════════════════════════════════
#  Row 6: All Stocks — Mini Charts
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 All Stocks — Mini Charts")

all_tickers_with_data = [t for t in SYSTEM.tickers if not data["bars"].get(t, pd.DataFrame()).empty]

if all_tickers_with_data:
    cols_per_row = 4
    for row_start in range(0, len(all_tickers_with_data), cols_per_row):
        batch = all_tickers_with_data[row_start:row_start + cols_per_row]
        cols = st.columns(len(batch))
        for col, tk in zip(cols, batch):
            with col:
                bdf = data["bars"][tk]
                last_close = bdf["close"].iloc[-1]
                first_close = bdf["close"].iloc[0]
                chg = last_close - first_close
                chg_pct = (chg / first_close * 100) if first_close != 0 else 0
                color = "#00cc96" if chg >= 0 else "#ef553b"

                st.markdown(
                    f"**{tk}** &nbsp; "
                    f"<span style='color:{color}'>₹{last_close:.2f} ({chg_pct:+.2f}%)</span>",
                    unsafe_allow_html=True,
                )
                fig_mini = go.Figure()
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fig_mini.add_trace(go.Scatter(
                    x=bdf["timestamp"], y=bdf["close"],
                    mode="lines", line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({r},{g},{b},0.08)",
                ))
                fig_mini.update_layout(
                    height=120, margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    template="plotly_dark",
                )
                st.plotly_chart(fig_mini, use_container_width=True, key=f"mini_{tk}")
else:
    st.info("No stock data to display")


# ═══════════════════════════════════════════════════════════════════════
#  Row 7: Active Positions + Sentiment Feed
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
col_pos, col_sent = st.columns(2)

with col_pos:
    st.subheader("Active Positions")
    pos_df = data["positions"]
    if not pos_df.empty:
        pnl_col = "Unrealized PnL" if "Unrealized PnL" in pos_df.columns else None
        if pnl_col:
            def _color_pnl(val):
                if isinstance(val, (int, float)):
                    color = "#00cc96" if val >= 0 else "#ef553b"
                    return f"color: {color}; font-weight: bold"
                return ""

            fmt = {}
            for c in ["Entry", "Current", "Unrealized PnL", "Stop Loss", "Take Profit"]:
                if c in pos_df.columns:
                    fmt[c] = "₹{:.2f}"

            styled = pos_df.style.map(_color_pnl, subset=[pnl_col]).format(fmt)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

with col_sent:
    st.subheader("📰 News Sentiment Feed")
    sent_df = data["sentiment_feed"]
    if not sent_df.empty:
        def _color_sentiment(val):
            colors = {"bullish": "#00cc96", "bearish": "#ef553b", "neutral": "#636efa"}
            return f"color: {colors.get(str(val).lower(), 'white')}; font-weight: bold"

        def _color_score(val):
            if isinstance(val, (int, float)):
                color = "#00cc96" if val > 0.3 else "#ef553b" if val < -0.3 else "#636efa"
                return f"color: {color}; font-weight: bold"
            return ""

        styled_sent = sent_df.style
        if "Sentiment" in sent_df.columns:
            styled_sent = styled_sent.map(_color_sentiment, subset=["Sentiment"])
        if "Score" in sent_df.columns:
            styled_sent = styled_sent.map(_color_score, subset=["Score"])
            styled_sent = styled_sent.format({"Score": "{:.2f}"})

        st.dataframe(styled_sent, use_container_width=True, hide_index=True, height=350)
    else:
        st.info("No sentiment data yet")


# ═══════════════════════════════════════════════════════════════════════
#  Row 8: Full Trade Log
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Complete Trade Log")
trade_df = data["trade_log"]
if not trade_df.empty:
    display_cols = [c for c in trade_df.columns if c not in ("trade_number", "cumulative_pnl", "result")]
    tl = trade_df[display_cols].copy() if display_cols else trade_df.copy()

    if "pnl" in tl.columns:
        def _color_trade_pnl(val):
            if isinstance(val, (int, float)):
                color = "#00cc96" if val >= 0 else "#ef553b"
                return f"color: {color}; font-weight: bold"
            return ""
        styled_tl = tl.style.map(_color_trade_pnl, subset=["pnl"]).format({"pnl": "₹{:+,.2f}"})
        st.dataframe(styled_tl, use_container_width=True, hide_index=True)
    else:
        st.dataframe(tl, use_container_width=True, hide_index=True)
else:
    st.info("No trades yet")


# ═══════════════════════════════════════════════════════════════════════
#  Row 9: Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🎯 Strategy Comparison")

by_strat = data.get("analytics_by_strategy", {})
if by_strat:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        strat_names = list(by_strat.keys())
        strat_pnls = [by_strat[s]["total_pnl"] for s in strat_names]
        strat_colors = ["#00cc96" if p >= 0 else "#ef553b" for p in strat_pnls]

        fig_strat = go.Figure(data=[go.Bar(
            x=strat_names, y=strat_pnls,
            marker_color=strat_colors,
            text=[f"₹{v:+,.0f}" for v in strat_pnls],
            textposition="outside",
        )])
        fig_strat.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="P&L (₹)", template="plotly_dark",
            title="Strategy P&L",
        )
        st.plotly_chart(fig_strat, use_container_width=True)

    with col_s2:
        strat_wr = [by_strat[s]["win_rate_pct"] for s in strat_names]
        strat_trades_count = [by_strat[s]["total_trades"] for s in strat_names]

        fig_strat_wr = go.Figure(data=[go.Bar(
            x=strat_names, y=strat_wr,
            marker_color=["#00cc96" if w >= 50 else "#ef553b" for w in strat_wr],
            text=[f"{w:.0f}% ({n}t)" for w, n in zip(strat_wr, strat_trades_count)],
            textposition="outside",
        )])
        fig_strat_wr.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.4)
        fig_strat_wr.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Win Rate (%)", yaxis_range=[0, 105],
            template="plotly_dark", title="Strategy Win Rates",
        )
        st.plotly_chart(fig_strat_wr, use_container_width=True)

    # Strategy detail table
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
else:
    st.info("Strategy comparison will appear after trades complete")


# ═══════════════════════════════════════════════════════════════════════
#  Auto-refresh
# ═══════════════════════════════════════════════════════════════════════
if auto_refresh:
    time.sleep(5)
    st.rerun()
