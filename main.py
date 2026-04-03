"""
Main Orchestrator — wires every module together and runs the live trading loop.

Usage:
    python main.py                # live trading (paper by default)
    python main.py --backtest     # run backtest over historical data
    python main.py --dashboard    # launch the Streamlit monitoring dashboard
"""

from __future__ import annotations

import argparse
import signal as os_signal
import subprocess
import sys
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
from execution.order_executor import OrderExecutor
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.sentiment_strategy import SentimentStrategy

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
        self.risk = RiskManager(initial_equity=10_00_000.0)  # ₹10 lakh
        self.executor = OrderExecutor(live=live)
        self.db = TradeDatabase()

        # Strategies
        self.strategies: List[BaseStrategy] = [
            MomentumStrategy(tickers=SYSTEM.tickers),
            MeanReversionStrategy(tickers=SYSTEM.tickers),
            SentimentStrategy(tickers=SYSTEM.tickers),
        ]

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
        except KeyboardInterrupt:
            log.info("Shutdown requested — stopping …")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.feed.stop()
        self.sentiment.stop()
        log.info("Trading engine stopped.")

    # ── event handlers ────────────────────────────────────────────────

    def _on_bar(self, bar: Bar) -> None:
        """Process a new price bar through the full pipeline."""
        exit_signals = self.risk.update_prices(bar)
        for sig in exit_signals:
            self.risk.close_position(sig.ticker, bar.close)

        if self.risk.is_halted:
            return
        for strat in self.strategies:
            signal = strat.on_bar(bar)
            if signal is None:
                continue
            self._process_signal(signal, bar.close)
    def _on_sentiment(self, result: SentimentResult) -> None:
        """Feed sentiment to strategies that react to it."""
        for strat in self.strategies:
            signal = strat.on_sentiment(result)
            if signal is None:
                continue
            latest = self.feed.latest_bars.get(result.ticker)
            price = latest.close if latest else 0.0
            if price > 0:
                self._process_signal(signal, price)
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
