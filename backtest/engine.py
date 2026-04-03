"""
Event-driven backtesting engine.

Replays historical bars through the same strategy → risk → execution pipeline
used in live trading, ensuring zero code divergence between backtest and live.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

import pandas as pd

from config.logging_config import get_logger
from core.models import Bar, Signal, SignalType, Side
from data_engine.historical_loader import fetch_bars, bars_to_models
from execution.order_executor import OrderExecutor
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy

log = get_logger(__name__)


class BacktestEngine:
    """
    Feeds historical data bar-by-bar into registered strategies,
    routes signals through risk management and the simulated executor,
    and collects performance metrics.
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        tickers: List[str],
        start: datetime,
        end: datetime,
        initial_equity: float = 10_00_000.0,
        timeframe: str = "1Min",
    ) -> None:
        self.strategies = strategies
        self.tickers = tickers
        self.start = start
        self.end = end
        self.timeframe = timeframe

        self.risk = RiskManager(initial_equity=initial_equity)
        self.executor = OrderExecutor(live=False)

        # Results
        self.equity_curve: List[Dict] = []
        self.all_signals: List[Signal] = []

    def run(self) -> pd.DataFrame:
        """Execute the backtest and return an equity-curve DataFrame."""
        log.info(
            "Backtest started — %s → %s | tickers=%s | strategies=%s",
            self.start.date(), self.end.date(),
            self.tickers, [s.name for s in self.strategies],
        )

        # 1. Load historical data
        bar_frames: Dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            df = fetch_bars(ticker, start=self.start, end=self.end)
            if not df.empty:
                bar_frames[ticker] = df

        if not bar_frames:
            log.error("No data loaded — aborting backtest")
            return pd.DataFrame()

        # 2. Merge all bars into a single time-ordered stream
        all_bars: List[Bar] = []
        for ticker, df in bar_frames.items():
            all_bars.extend(bars_to_models(df, ticker))
        all_bars.sort(key=lambda b: b.timestamp)

        # 3. Replay
        for bar in all_bars:
            self._process_bar(bar)

        # 4. Close remaining positions at last known prices
        for ticker in list(self.risk.positions):
            last_bar = bar_frames[ticker].iloc[-1]
            self.risk.close_position(ticker, float(last_bar["close"]))

        log.info(
            "Backtest complete — final equity INR %.2f  trades=%d",
            self.risk.equity, len(self.risk.closed_trades),
        )

        return pd.DataFrame(self.equity_curve)

    # ── internals ─────────────────────────────────────────────────────

    def _process_bar(self, bar: Bar) -> None:
        # a) Risk: update prices & check SL/TP
        exit_signals = self.risk.update_prices(bar)
        for sig in exit_signals:
            self._execute_exit(sig, bar.close)

        # b) Strategies: generate entry/exit signals
        for strat in self.strategies:
            signal = strat.on_bar(bar)
            if signal is None:
                continue
            self.all_signals.append(signal)

            if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
                self._execute_entry(signal, bar.close)
            elif signal.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
                self._execute_exit(signal, bar.close)

        # c) Record equity snapshot
        snap = self.risk.snapshot()
        self.equity_curve.append({
            "timestamp": bar.timestamp,
            "equity": snap.equity,
            "cash": snap.cash,
            "drawdown_pct": snap.drawdown_pct,
        })

    def _execute_entry(self, signal: Signal, price: float) -> None:
        if not self.risk.approve_signal(signal, price):
            return
        qty = self.risk.compute_position_size(signal, price)
        if qty <= 0:
            return
        order = self.executor.execute(signal, qty, price)
        if order and order.filled_price:
            self.risk.open_position(order)

    def _execute_exit(self, signal: Signal, price: float) -> None:
        self.risk.close_position(signal.ticker, price)
