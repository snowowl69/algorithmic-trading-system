"""
Corporate Actions Pipeline — fetches upcoming and past dividend data,
including historical price impact (before/after ex-date) for dashboard display.
"""

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import yfinance as yf
import pandas as pd

from config.logging_config import get_logger
from config.settings import SYSTEM
from data_engine.historical_loader import _YFINANCE_MAP

log = get_logger(__name__)


class CorporateActionsPipeline:
    """
    Background thread that fetches dividend data for all tickers,
    analyzes price impact of the last dividend, and caches the results.
    """

    def __init__(self, tickers: Optional[List[str]] = None) -> None:
        if tickers:
            self.tickers = tickers
        else:
            # Broaden the target list for dividends
            from data_engine.news_feed import NIFTY_50, NIFTY_NEXT_50
            combined = set(SYSTEM.tickers + NIFTY_50 + NIFTY_NEXT_50)
            self.tickers = list(combined)
            
        self.dividend_results: List[Dict] = []
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="corporate-actions"
        )
        self._thread.start()
        log.info("CorporateActionsPipeline started.")

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                self._fetch_all()
            except Exception:
                log.exception("Error in CorporateActionsPipeline")
            
            for _ in range(12 * 60 * 60 // 5):
                if not self._running.is_set():
                    return
                time.sleep(5)

    def _fetch_all(self) -> None:
        results = []
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_ticker = {executor.submit(self._fetch_for_ticker, tk): tk for tk in self.tickers if self._running.is_set()}
            for future in concurrent.futures.as_completed(future_to_ticker):
                tk = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception as e:
                    log.debug("Corporate actions fetch failed for %s: %s", tk, e)
        
        if results:
            self.dividend_results = results
            log.info("Corporate actions fetched for %d tickers", len(results))

    def _fetch_for_ticker(self, ticker: str) -> Optional[Dict]:
        yf_sym = _YFINANCE_MAP.get(ticker, f"{ticker}.NS")
        tk = yf.Ticker(yf_sym)
        
        # Get historical dividends
        divs = tk.dividends
        if divs.empty:
            return None
            
        last_div_date = divs.index[-1]
        last_div_amount = float(divs.iloc[-1])
        
        # Analyze price impact of the last dividend
        # Fetch data starting 7 days before to 7 days after to guarantee trading days
        start_date = last_div_date - timedelta(days=7)
        end_date = last_div_date + timedelta(days=7)
        
        hist = tk.history(start=start_date, end=end_date, interval="1d")
        
        price_before = 0.0
        price_after = 0.0
        pct_change = 0.0
        
        if not hist.empty:
            # Price before ex-date (the day before)
            before_df = hist[hist.index < last_div_date]
            if not before_df.empty:
                price_before = float(before_df['Close'].iloc[-1])
                
            # Price on ex-date (or after)
            after_df = hist[hist.index >= last_div_date]
            if not after_df.empty:
                price_after = float(after_df['Close'].iloc[0])
                
            if price_before > 0:
                pct_change = ((price_after - price_before) / price_before) * 100
        
        # Get upcoming corporate actions (Dividends & Earnings)
        upcoming_div = "N/A"
        upcoming_earnings = "N/A"
        try:
            calendar = tk.calendar
            if isinstance(calendar, dict):
                if 'Ex-Dividend Date' in calendar:
                    dt = calendar['Ex-Dividend Date']
                    if pd.notna(dt):
                        upcoming_div = dt.strftime("%Y-%m-%d")
                if 'Earnings Date' in calendar:
                    ed = calendar['Earnings Date']
                    if isinstance(ed, list) and len(ed) > 0 and pd.notna(ed[0]):
                        upcoming_earnings = ed[0].strftime("%Y-%m-%d")
        except Exception:
            pass

        return {
            "Ticker": ticker,
            "Last Div Date": last_div_date.strftime("%Y-%m-%d"),
            "Last Div Amount": f"₹{last_div_amount:.2f}",
            "Price Before": f"₹{price_before:.2f}" if price_before else "-",
            "Price On Ex-Date": f"₹{price_after:.2f}" if price_after else "-",
            "Impact": f"{pct_change:+.2f}%" if price_before else "-",
            "Upcoming Div Ex-Date": upcoming_div,
            "Upcoming Earnings": upcoming_earnings,
        }
