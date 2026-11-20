"""
Live Options Data Feed & Option Chain Streamer for Indian Derivatives.
Provides real-time Greeks, Open Interest (OI) analysis, Max Pain, and multi-expiry chains.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from analytics.options_engine import OptionsEngine, OptionChainSnapshot
from config.logging_config import get_logger

log = get_logger(__name__)


# Default benchmark reference prices for Indian Derivatives
_DEFAULT_OPTIONS_UNDERLYINGS = {
    "NIFTY 50": 24500.0,
    "BANKNIFTY": 52100.0,
    "FINNIFTY": 23400.0,
    "RELIANCE": 2980.0,
    "HDFCBANK": 1720.0,
    "TCS": 3950.0,
    "INFY": 1820.0,
}


class LiveOptionsFeed:
    """
    Sub-system managing real-time option chain computations and streaming.
    """

    def __init__(self, engine: Optional[OptionsEngine] = None) -> None:
        self.engine = engine or OptionsEngine()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Spot price tracking
        self.spot_prices: Dict[str, float] = dict(_DEFAULT_OPTIONS_UNDERLYINGS)
        # Cached chain snapshots
        self.chains: Dict[str, Dict[str, Any]] = {}
        
        # Initialize initial option chains
        self._refresh_all_chains()

    def update_spot(self, underlying: str, spot: float) -> None:
        """Update spot price and trigger Greek recalculation."""
        if spot > 0:
            with self._lock:
                self.spot_prices[underlying] = spot
                self._update_chain_for(underlying, spot)

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True, name="options-feed")
        self._thread.start()
        log.info("LiveOptionsFeed started for %d assets", len(self.spot_prices))

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3)
        log.info("LiveOptionsFeed stopped")

    def _stream_loop(self) -> None:
        """Background loop updating option chains with time decay and small volatility jitter."""
        import random
        while self._running.is_set():
            try:
                with self._lock:
                    for underlying, spot in list(self.spot_prices.items()):
                        # Subtle realistic high-frequency tick simulation
                        jitter = spot * random.uniform(-0.0008, 0.0008)
                        new_spot = round(spot + jitter, 2)
                        self.spot_prices[underlying] = new_spot
                        self._update_chain_for(underlying, new_spot)
            except Exception as e:
                log.debug("Options feed loop error: %s", e)

            # Update option chains every 2 seconds
            for _ in range(20):
                if not self._running.is_set():
                    return
                time.sleep(0.1)

    def _update_chain_for(self, underlying: str, spot: float) -> None:
        """Compute option chain and format as dictionary."""
        chain = self.engine.generate_option_chain(
            underlying=underlying,
            spot_price=spot,
            atm_iv=0.145 if "NIFTY" in underlying else 0.22,
            dte_days=4.0,  # Weekly expiry
            num_strikes=17,
        )
        
        # Convert to serializable dict
        strikes_data = []
        for s in chain.strikes:
            strikes_data.append({
                "strike": s.strike,
                "call_ltp": s.call_ltp,
                "call_iv": s.call_iv,
                "call_delta": s.call_delta,
                "call_gamma": s.call_gamma,
                "call_theta": s.call_theta,
                "call_vega": s.call_vega,
                "call_oi": s.call_oi,
                "call_oi_change": s.call_oi_change,
                "call_volume": s.call_volume,
                "put_ltp": s.put_ltp,
                "put_iv": s.put_iv,
                "put_delta": s.put_delta,
                "put_gamma": s.put_gamma,
                "put_theta": s.put_theta,
                "put_vega": s.put_vega,
                "put_oi": s.put_oi,
                "put_oi_change": s.put_oi_change,
                "put_volume": s.put_volume,
            })

        self.chains[underlying] = {
            "underlying": chain.underlying,
            "spot_price": chain.spot_price,
            "atm_strike": chain.atm_strike,
            "expiry_date": chain.expiry_date,
            "dte_days": chain.dte_days,
            "pcr_oi": chain.pcr_oi,
            "pcr_volume": chain.pcr_volume,
            "max_pain_strike": chain.max_pain_strike,
            "atm_iv": chain.atm_iv,
            "iv_rank": chain.iv_rank,
            "iv_percentile": chain.iv_percentile,
            "strikes": strikes_data,
            "timestamp": chain.timestamp,
        }

    def _refresh_all_chains(self) -> None:
        for underlying, spot in self.spot_prices.items():
            self._update_chain_for(underlying, spot)

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns thread-safe snapshot of all option chains."""
        with self._lock:
            return dict(self.chains)
