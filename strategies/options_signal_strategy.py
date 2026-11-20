"""
Options-Signal Strategy.

Generates contrarian signals based on the Put-Call Ratio (PCR) and
IV Rank derived from an options chain state file.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from core.models import Bar, SentimentResult, Signal, SignalType
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# Path to the live-state file containing options chain data
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "live_state.json",
)


class OptionsSignalStrategy(BaseStrategy):
    """Generate signals from options-market data (PCR and IV Rank).

    The strategy reads ``data/live_state.json`` for an ``options_chains``
    key.  When present it applies contrarian logic:

    * **PCR > 1.3 and rising** -- heavy put buying suggests fear; contrarian
      bullish -> ``LONG_ENTRY``.
    * **PCR < 0.7** -- heavy call buying suggests complacency; contrarian
      bearish -> ``SHORT_ENTRY``.
    * **IV Rank > 80** -- premiums are elevated; avoid new entries.

    If no options data is available the strategy returns ``None``.
    """

    name = "options_signal"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._prev_pcr: Dict[str, float] = {}

    # ---- core logic ---------------------------------------------------

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        """Check options data and return a signal if PCR conditions are met."""
        self._record_bar(bar)
        ticker = bar.ticker

        options_data = self._load_options_data(ticker)
        if options_data is None:
            return None

        pcr = options_data.get("pcr")
        iv_rank = options_data.get("iv_rank")

        if pcr is None:
            return None

        # IV Rank filter: avoid new entries when premiums are extreme
        if iv_rank is not None and iv_rank > 80:
            self._prev_pcr[ticker] = pcr
            return None

        prev_pcr = self._prev_pcr.get(ticker)
        self._prev_pcr[ticker] = pcr

        # Contrarian bullish: heavy put buying (PCR > 1.3 and rising)
        if pcr > 1.3 and prev_pcr is not None and pcr > prev_pcr:
            return Signal(
                ticker=ticker,
                signal_type=SignalType.LONG_ENTRY,
                strategy_name=self.name,
                strength=min(0.5 + (pcr - 1.3) * 0.5, 1.0),
                metadata={"pcr": pcr, "iv_rank": iv_rank, "reason": "high_pcr_rising"},
            )

        # Contrarian bearish: heavy call buying (PCR < 0.7)
        if pcr < 0.7:
            return Signal(
                ticker=ticker,
                signal_type=SignalType.SHORT_ENTRY,
                strategy_name=self.name,
                strength=min(0.5 + (0.7 - pcr) * 0.5, 1.0),
                metadata={"pcr": pcr, "iv_rank": iv_rank, "reason": "low_pcr"},
            )

        return None

    def on_sentiment(self, result: SentimentResult) -> Optional[Signal]:
        """Options strategy does not react to sentiment data."""
        return None

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _load_options_data(ticker: str) -> Optional[Dict[str, Any]]:
        """Try to read options-chain data from the state file.

        Returns a dict with ``pcr`` and ``iv_rank`` keys for the given
        ticker, or ``None`` if the data is unavailable.
        """
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        chains = state.get("options_chains")
        if not isinstance(chains, dict):
            return None

        ticker_data = chains.get(ticker) or chains.get(ticker.upper())
        if not isinstance(ticker_data, dict):
            return None

        return ticker_data
