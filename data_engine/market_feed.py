"""
Live market data feed for Indian markets (NSE / BSE / MCX).

Uses a polling-based approach with yfinance for real-time-ish data.
On each new bar the feed publishes a ``Bar`` object onto the event bus
channel ``"market:bar"``.

For true real-time feeds, plug in broker WebSocket APIs (Angel One, Upstox,
Zerodha Kite Connect) by subclassing ``BaseFeed``.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.logging_config import get_logger
from config.settings import SYSTEM
from core.events import bus
from core.models import Bar
from analytics.candle_maker import LiveCandleMaker, LiveTick

log = get_logger(__name__)


class IndianMarketFeed:
    """
    Polls yfinance at regular intervals for NSE/BSE/MCX tickers.

    For demo / paper trading.  Replace with broker WebSocket for
    production live trading.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        poll_interval: int = 15,
    ) -> None:
        self.tickers = tickers or SYSTEM.tickers
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._latest_bars: Dict[str, Bar] = {}
        self.prev_closes: Dict[str, float] = {}
        self.candle_maker = LiveCandleMaker()

    def get_forming_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Return real-time forming candle snapshots for all active tickers."""
        res = {}
        for t in self.tickers:
            fc = self.candle_maker.get_forming_candle(t)
            if fc:
                res[t] = {
                    "open": fc.open,
                    "high": fc.high,
                    "low": fc.low,
                    "close": fc.close,
                    "volume": fc.volume,
                    "buy_volume": fc.buy_volume,
                    "sell_volume": fc.sell_volume,
                    "delta": fc.delta,
                    "tick_count": fc.tick_count,
                    "open_time": fc.open_time.isoformat(),
                }
        return res

    def start(self) -> None:
        if self._running.is_set():
            log.warning("IndianMarketFeed already running")
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="india-feed"
        )
        self._thread.start()
        log.info("IndianMarketFeed started for %d tickers", len(self.tickers))

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("IndianMarketFeed stopped")

    @property
    def latest_bars(self) -> Dict[str, Bar]:
        return dict(self._latest_bars)

    def _poll_loop(self) -> None:
        while self._running.is_set():
            try:
                self._fetch_latest()
            except Exception:
                log.exception("IndianMarketFeed poll error")
            # Wait poll_interval, but check _running every second
            for _ in range(self.poll_interval):
                if not self._running.is_set():
                    return
                time.sleep(1)

    def _fetch_latest(self) -> None:
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed — feed idle")
            return

        from data_engine.historical_loader import _YFINANCE_MAP

        def fetch_ticker(ticker: str) -> None:
            if not self._running.is_set():
                return
            try:
                yf_sym = _YFINANCE_MAP.get(ticker, f"{ticker}.NS")
                tk = yf.Ticker(yf_sym)
                
                # Fetch prev_close on first run
                if ticker not in self.prev_closes:
                    try:
                        self.prev_closes[ticker] = float(tk.fast_info['previous_close'])
                    except Exception as e:
                        log.debug("Could not get previous_close for %s: %s", ticker, e)
                        self.prev_closes[ticker] = 0.0

                hist = tk.history(period="1d", interval="1m")
                if hist.empty:
                    # Off-hours / weekend / rate-limit fallback: bridge with realistic tick simulation
                    base_px = self.prev_closes.get(ticker, 0.0)
                    if base_px <= 0:
                        from data_engine.historical_loader import _INR_BASE_PRICES
                        base_px = _INR_BASE_PRICES.get(ticker, 1500.0)
                    import random
                    jitter = base_px * random.uniform(-0.0015, 0.0015)
                    sim_close = round(base_px + jitter, 2)
                    bar = Bar(
                        ticker=ticker,
                        timestamp=datetime.now(timezone.utc),
                        open=round(sim_close * random.uniform(0.999, 1.001), 2),
                        high=round(sim_close * random.uniform(1.000, 1.002), 2),
                        low=round(sim_close * random.uniform(0.998, 1.000), 2),
                        close=sim_close,
                        volume=float(random.randint(500, 15000)),
                    )
                    self._latest_bars[ticker] = bar
                    self.candle_maker.ingest_tick(
                        LiveTick(
                            ticker=ticker,
                            price=bar.close,
                            volume=bar.volume,
                            timestamp=bar.timestamp,
                            is_buyer_maker=(bar.close < bar.open),
                        )
                    )
                    bus.publish("market:bar", bar)
                    log.info("Simulated Bar >> %s INR %.2f vol=%d", ticker, bar.close, bar.volume)
                    return
                last = hist.iloc[-1]
                # Use exchange timestamp when available, fall back to local time
                try:
                    exchange_ts = hist.index[-1].to_pydatetime()
                    if exchange_ts.tzinfo is None:
                        exchange_ts = exchange_ts.replace(tzinfo=timezone.utc)
                except Exception:
                    exchange_ts = datetime.now(timezone.utc)
                bar = Bar(
                    ticker=ticker,
                    timestamp=exchange_ts,
                    open=float(last.get("Open", last.get("Close", 0))),
                    high=float(last.get("High", last.get("Close", 0))),
                    low=float(last.get("Low", last.get("Close", 0))),
                    close=float(last["Close"]),
                    volume=float(last.get("Volume", 0)),
                )
                self._latest_bars[ticker] = bar
                self.candle_maker.ingest_tick(
                    LiveTick(
                        ticker=ticker,
                        price=bar.close,
                        volume=bar.volume,
                        timestamp=bar.timestamp,
                        is_buyer_maker=(bar.close < bar.open),
                    )
                )
                bus.publish("market:bar", bar)
                log.info("Bar >> %s INR %.2f vol=%d", ticker, bar.close, bar.volume)
            except Exception as exc:
                log.warning("Failed to fetch %s: %s (using offline bridge)", ticker, exc)
                # Network error fallback
                from data_engine.historical_loader import _INR_BASE_PRICES
                base_px = self.prev_closes.get(ticker) or _INR_BASE_PRICES.get(ticker, 1500.0)
                import random
                sim_close = round(base_px * (1.0 + random.uniform(-0.001, 0.001)), 2)
                bar = Bar(
                    ticker=ticker,
                    timestamp=datetime.now(timezone.utc),
                    open=sim_close, high=round(sim_close * 1.001, 2),
                    low=round(sim_close * 0.999, 2), close=sim_close,
                    volume=1000.0,
                )
                self._latest_bars[ticker] = bar
                self.candle_maker.ingest_tick(
                    LiveTick(
                        ticker=ticker,
                        price=bar.close,
                        volume=bar.volume,
                        timestamp=bar.timestamp,
                        is_buyer_maker=(bar.close < bar.open),
                    )
                )
                bus.publish("market:bar", bar)

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(fetch_ticker, t) for t in self.tickers]
            concurrent.futures.wait(futures)
        fetched = len(self._latest_bars)
        log.info("Feed poll complete — %d / %d tickers updated", fetched, len(self.tickers))
        
        # Check for dynamic tickers added from the dashboard
        try:
            import os
            txt_path = os.path.join("data", "dynamic_tickers.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    new_tickers = [line.strip() for line in f if line.strip()]
                for t in new_tickers:
                    if t not in self.tickers:
                        self.tickers.append(t)
                        log.info("Dynamically added ticker to engine: %s", t)
                # clear file so we don't read it endlessly
                open(txt_path, "w").close()
        except Exception as e:
            log.debug("Failed to read dynamic tickers: %s", e)


class SyntheticFeed:
    """
    Generates synthetic bars for demo / offline mode.
    Replays from the synthetic data generator at realistic speed.
    """

    def __init__(self, tickers: Optional[List[str]] = None) -> None:
        self.tickers = tickers or SYSTEM.tickers
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._latest_bars: Dict[str, Bar] = {}
        self.prev_closes: Dict[str, float] = {}
        self.candle_maker = LiveCandleMaker()

    def get_forming_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Return real-time forming candle snapshots for all active tickers."""
        res = {}
        for t in self.tickers:
            fc = self.candle_maker.get_forming_candle(t)
            if fc:
                res[t] = {
                    "open": fc.open,
                    "high": fc.high,
                    "low": fc.low,
                    "close": fc.close,
                    "volume": fc.volume,
                    "buy_volume": fc.buy_volume,
                    "sell_volume": fc.sell_volume,
                    "delta": fc.delta,
                    "tick_count": fc.tick_count,
                    "open_time": fc.open_time.isoformat(),
                }
        return res

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._generate_loop, daemon=True, name="synthetic-feed"
        )
        self._thread.start()
        log.info("SyntheticFeed started for %d tickers", len(self.tickers))

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("SyntheticFeed stopped")

    @property
    def latest_bars(self) -> Dict[str, Bar]:
        return dict(self._latest_bars)

    def _generate_loop(self) -> None:
        import numpy as np
        from data_engine.historical_loader import _INR_BASE_PRICES

        rng = np.random.default_rng(seed=42)
        prices = {t: _INR_BASE_PRICES.get(t, 1500.0) for t in self.tickers}

        while self._running.is_set():
            for ticker in self.tickers:
                if not self._running.is_set():
                    return
                price = prices[ticker]
                change = price * 0.001 * rng.standard_normal()
                price = max(price + change, 1.0)
                prices[ticker] = price

                bar = Bar(
                    ticker=ticker,
                    timestamp=datetime.now(timezone.utc),
                    open=round(price * (1 + 0.001 * rng.standard_normal()), 2),
                    high=round(price * (1 + abs(0.002 * rng.standard_normal())), 2),
                    low=round(price * (1 - abs(0.002 * rng.standard_normal())), 2),
                    close=round(price, 2),
                    volume=float(int(rng.exponential(10000) + 100)),
                )
                self._latest_bars[ticker] = bar
                self.candle_maker.ingest_tick(
                    LiveTick(
                        ticker=ticker,
                        price=bar.close,
                        volume=bar.volume,
                        timestamp=bar.timestamp,
                        is_buyer_maker=(bar.close < bar.open),
                    )
                )
                bus.publish("market:bar", bar)

            time.sleep(1)


def create_feed(provider: Optional[str] = None):
    """Return the appropriate feed based on configuration."""
    provider = (provider or SYSTEM.data_provider).lower()
    if provider == "yfinance":
        return IndianMarketFeed()
    return SyntheticFeed()
