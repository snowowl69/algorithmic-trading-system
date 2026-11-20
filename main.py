"""
Main Orchestrator — wires every module together and runs the live trading loop.

Usage:
    python main.py                # live trading (paper by default)
    python main.py --backtest     # run backtest over historical data
    python main.py --dashboard    # launch the Streamlit monitoring dashboard
"""

from __future__ import annotations

import argparse
import concurrent.futures
import signal as os_signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Ensure project root is importable
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.logging_config import get_logger
from config.settings import SYSTEM
from core.events import bus
from core.models import Bar, SentimentResult, Signal, SignalType
from data_engine.market_feed import create_feed
from data_engine.nlp_sentiment import SentimentPipeline
from data_engine.database import TradeDatabase
from data_engine.corporate_actions import CorporateActionsPipeline
from data_engine.options_feed import LiveOptionsFeed
from execution.order_executor import OrderExecutor
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.sentiment_strategy import SentimentStrategy
from strategies.vwap_strategy import VWAPStrategy
from strategies.options_signal_strategy import OptionsSignalStrategy
from analytics.market_regime import MarketRegimeDetector
from analytics.session_manager import get_session_info
from analytics.trade_recommender import TradeRecommender

log = get_logger("main")
class TradingEngine:
    """
    Central coordinator that:
        1. Starts the market data feed & sentiment pipeline.
        2. Subscribes to event bus channels.
        3. Routes bars → strategies → risk check → execution.
    """

    def __init__(self, live: bool = False) -> None:
        self.live = live

        # Sub-systems
        self.feed = create_feed()
        self.sentiment = SentimentPipeline()
        self.corporate_actions = CorporateActionsPipeline()
        self.options_feed = LiveOptionsFeed()
        self.risk = RiskManager(initial_equity=10_00_000.0)  # ₹10 lakh
        self.executor = OrderExecutor(live=live)
        self.db = TradeDatabase()
        self.regime_detector = MarketRegimeDetector()
        self.trade_recommender = TradeRecommender()
        
        # Thread pool for processing signals and state concurrently
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        # Strategies
        self.strategies: List[BaseStrategy] = [
            MomentumStrategy(tickers=SYSTEM.tickers),
            MeanReversionStrategy(tickers=SYSTEM.tickers),
            SentimentStrategy(tickers=SYSTEM.tickers),
            VWAPStrategy(tickers=SYSTEM.tickers),
            OptionsSignalStrategy(tickers=SYSTEM.tickers),
        ]

        # Dashboard State Tracking
        self._bars_history: Dict[str, List[Dict]] = {t: [] for t in SYSTEM.tickers}
        self._equity_curve: List[Dict] = []
        self._sentiment_feed: List[Dict] = []

        self._running = True

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        log.info("=" * 60)
        log.info("  ALGO TRADING ENGINE — %s MODE", "LIVE" if self.live else "PAPER")
        log.info("  Tickers: %s", SYSTEM.tickers)
        log.info("  Strategies: %s", [s.name for s in self.strategies])
        log.info("=" * 60)
        bus.subscribe("market:bar", self._on_bar)
        bus.subscribe("nlp:sentiment", self._on_sentiment)
        bus.subscribe("risk:circuit_breaker", self._on_circuit_breaker)
        self.feed.start()
        self.sentiment.start()
        self.corporate_actions.start()
        self.options_feed.start()
        self.risk.reset_daily()
        try:
            while self._running:
                time.sleep(SYSTEM.heartbeat_interval_sec)
                snap = self.risk.snapshot()
                log.info(
                    "HEARTBEAT | equity=INR %.2f  pnl=INR %.2f (%.2f%%)  "
                    "positions=%d  halted=%s",
                    snap.equity, snap.daily_pnl, snap.daily_pnl_pct,
                    snap.open_positions, self.risk.is_halted,
                )
                
                # ── Ingest Manual Trades ──
                import os, json
                manual_trades_path = os.path.join("data", "manual_trades.json")
                if os.path.exists(manual_trades_path):
                    try:
                        with open(manual_trades_path, "r", encoding="utf-8") as f:
                            mtrades = json.load(f)
                        for mt in mtrades:
                            qty = mt.get("qty", 0)
                            entry = mt.get("entry", 0.0)
                            exit_px = mt.get("exit", 0.0)
                            side = mt.get("side", "buy").upper()
                            direction = 1.0 if side == "BUY" else -1.0
                            pnl = (exit_px - entry) * qty * direction if exit_px > 0 else 0.0
                            
                            self.risk.closed_trades.append({
                                "ticker": mt.get("ticker", "UNKNOWN"),
                                "side": side,
                                "entry": entry,
                                "exit": exit_px,
                                "qty": qty,
                                "pnl": round(pnl, 2),
                                "strategy": "MANUAL",
                                "closed_at": mt.get("timestamp", datetime.now(timezone.utc).isoformat())
                            })
                            # Adjust equity for manual trade PnL
                            self.risk.equity += pnl
                            log.info("Ingested Manual Trade: %s %s %d @ %.2f (PnL: %.2f)", side, mt.get("ticker"), qty, entry, pnl)
                        
                        # Clear the file after ingestion
                        os.remove(manual_trades_path)
                    except Exception as e:
                        log.error("Failed to ingest manual trades: %s", e)
                # ── Crash recovery: persist state to DB on every heartbeat ──
                # Run persistence in background thread to avoid blocking trading loop
                self._pool.submit(self._persist_state, snap)
        except KeyboardInterrupt:
            log.info("Shutdown requested — stopping …")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.feed.stop()
        self.sentiment.stop()
        self.corporate_actions.stop()
        self.options_feed.stop()
        self._pool.shutdown(wait=False)
        # Final state save on shutdown
        try:
            snap = self.risk.snapshot()
            self._persist_state(snap)
            if self.risk.closed_trades:
                self.db.insert_trades_bulk(self.risk.closed_trades)
                log.info("Saved %d trades to database on shutdown", len(self.risk.closed_trades))
        except Exception:
            log.exception("Failed to persist final state on shutdown")
        self.db.close()
        log.info("Trading engine stopped.")

    # ── state persistence ──────────────────────────────────────────────────

    def _persist_state(self, snap) -> None:
        """Save equity snapshot to DB and publish state for the dashboard."""
        try:
            self.db.insert_equity_snapshot(
                equity=snap.equity, cash=snap.cash, pnl=snap.daily_pnl
            )
            
            # ── Dashboard State Publishing ──
            self._equity_curve.append({
                "timestamp": snap.timestamp.isoformat(),
                "equity": snap.equity,
            })
            if len(self._equity_curve) > 500:
                self._equity_curve.pop(0)
                
            positions = []
            for p in self.risk.positions.values():
                positions.append({
                    "Ticker": p.ticker,
                    "Side": p.side.value.upper(),
                    "Qty": p.quantity,
                    "Entry": p.avg_entry_price,
                    "Current": p.current_price,
                    "Unrealized PnL": p.unrealized_pnl,
                    "Stop Loss": p.stop_loss,
                    "Take Profit": p.take_profit,
                    "Strategy": p.strategy_name
                })
                
            import dataclasses
            snap_dict = dataclasses.asdict(snap)
            snap_dict["halted"] = self.risk.is_halted
            
            regime_info = {}
            primary_ticker = "RELIANCE" if "RELIANCE" in self._bars_history else (SYSTEM.tickers[0] if SYSTEM.tickers else "")
            if primary_ticker and primary_ticker in self._bars_history and len(self._bars_history[primary_ticker]) >= 20:
                import numpy as np
                hist = self._bars_history[primary_ticker]
                closes = np.array([b["close"] for b in hist])
                highs = np.array([b["high"] for b in hist])
                lows = np.array([b["low"] for b in hist])
                res = self.regime_detector.detect(closes, highs, lows)
                regime_info = {
                    "regime": res["regime"].value if hasattr(res["regime"], "value") else str(res["regime"]),
                    "adx_value": res.get("adx_value", 0.0),
                    "atr_pct": res.get("atr_pct", 0.0),
                    "bb_width": res.get("bb_width", 0.0),
                    "squeeze": res.get("squeeze", False),
                    "ticker": primary_ticker,
                }

            session_data = get_session_info()
            session_info = {
                "session": session_data["session"].value if hasattr(session_data["session"], "value") else str(session_data["session"]),
                "label": session_data.get("label", ""),
                "start_time": session_data.get("start_time", ""),
                "end_time": session_data.get("end_time", ""),
                "minutes_remaining": session_data.get("minutes_remaining", 0),
                "should_avoid_entry": session_data.get("should_avoid_entry", False),
                "is_orb_window": session_data.get("is_orb_window", False),
            }

            trade_suggestions = self.trade_recommender.generate_recommendations(
                bars_dict=self._bars_history,
                options_chains=self.options_feed.get_snapshot(),
                regime_info=regime_info,
                prev_closes=getattr(self.feed, 'prev_closes', {}),
                sentiment_scores={
                    s.get("ticker", ""): s.get("score", 0.0)
                    for s in self._sentiment_feed[-20:] if isinstance(s, dict)
                },
                account_equity=snap.equity,
            )

            from core.state_store import publish_snapshot
            publish_snapshot(
                snapshot=snap_dict,
                positions=positions,
                bars=self._bars_history,
                sentiment_feed=self._sentiment_feed,
                trade_log=self.risk.closed_trades,
                equity_curve=self._equity_curve,
                prev_closes=getattr(self.feed, 'prev_closes', {}),
                dividends=self.corporate_actions.dividend_results,
                options_chains=self.options_feed.get_snapshot(),
                regime_info=regime_info,
                session_info=session_info,
                forming_candles=getattr(self.feed, 'get_forming_snapshots', lambda: {})(),
                trade_suggestions=trade_suggestions,
            )
        except Exception:
            log.exception("Failed to persist state snapshot")

    # ── event handlers ────────────────────────────────────────────────

    def _on_bar(self, bar: Bar) -> None:
        """Process a new price bar through the full pipeline."""
        # ── Track for Dashboard ──
        bar_dict = {
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        hist = self._bars_history[bar.ticker]
        if hist and hist[-1]["timestamp"] == bar_dict["timestamp"]:
            hist[-1] = bar_dict
        else:
            hist.append(bar_dict)
            
        if len(hist) > 500:
            hist.pop(0)

        # Update options desk spot pricing
        self.options_feed.update_spot(bar.ticker, bar.close)

        exit_signals = self.risk.update_prices(bar)
        for sig in exit_signals:
            self.risk.close_position(sig.ticker, bar.close)

        if self.risk.is_halted:
            return
            
        def eval_strat(strat, _bar):
            signal = strat.on_bar(_bar)
            if signal:
                self._process_signal(signal, _bar.close)

        futures = [self._pool.submit(eval_strat, strat, bar) for strat in self.strategies]
        # We don't block on futures; allow them to execute and route orders independently.
        
    def _on_sentiment(self, result: SentimentResult) -> None:
        """Feed sentiment to strategies that react to it."""
        # ── Track for Dashboard ──
        self._sentiment_feed.append({
            "Time": result.published_at.strftime("%H:%M"),
            "Ticker": result.ticker,
            "Headline": result.headline,
            "Source": result.source,
            "Sentiment": result.sentiment.value,
            "Score": result.score,
        })
        if len(self._sentiment_feed) > 50:
            self._sentiment_feed.pop(0)

        def eval_sentiment_strat(strat, _result):
            signal = strat.on_sentiment(_result)
            if signal:
                latest = self.feed.latest_bars.get(_result.ticker)
                price = latest.close if latest else 0.0
                if price > 0:
                    self._process_signal(signal, price)

        for strat in self.strategies:
            self._pool.submit(eval_sentiment_strat, strat, result)
            
    def _on_circuit_breaker(self, reason: str) -> None:
        log.critical("CIRCUIT BREAKER → %s", reason)
    def _process_signal(self, signal: Signal, price: float) -> None:
        """Route a validated signal through risk → execution."""
        if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
            if not self.risk.approve_signal(signal, price):
                return
            qty = self.risk.compute_position_size(signal, price)
            if qty <= 0:
                return
            order = self.executor.execute(signal, qty, price)
            if order and order.filled_price:
                self.risk.open_position(order)
        elif signal.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
            self.risk.close_position(signal.ticker, price)
def main() -> None:
    parser = argparse.ArgumentParser(description="Algo Trading System")
    parser.add_argument("--live", action="store_true", help="Use live broker API")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--days", type=int, default=30, help="Backtest lookback days")
    args = parser.parse_args()

    if args.dashboard:
        app_path = Path(__file__).parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
        return

    if args.backtest:
        from backtest.engine import BacktestEngine

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        strategies = [
            MomentumStrategy(tickers=SYSTEM.tickers),
            MeanReversionStrategy(tickers=SYSTEM.tickers),
        ]
        bt = BacktestEngine(
            strategies=strategies,
            tickers=SYSTEM.tickers,
            start=start, end=end,
        )
        eq_df = bt.run()
        if not eq_df.empty:
            print(f"\nFinal equity : INR {bt.risk.equity:,.2f}")
            print(f"Total trades : {len(bt.risk.closed_trades)}")
            print(f"Peak equity  : INR {bt.risk.peak_equity:,.2f}")
            snap = bt.risk.snapshot()
            print(f"Max drawdown : {snap.drawdown_pct:.2f}%")

            # Store all trades in database
            if bt.risk.closed_trades:
                db = TradeDatabase()
                db.insert_trades_bulk(bt.risk.closed_trades)
                print(f"Trades saved : {db.trade_count()} in database")
                db.close()
        return

    # Default: live / paper trading
    engine = TradingEngine(live=args.live)

    def _sigint_handler(sig, frame):
        engine.stop()
        sys.exit(0)

    os_signal.signal(os_signal.SIGINT, _sigint_handler)
    engine.start()


if __name__ == "__main__":
    main()
