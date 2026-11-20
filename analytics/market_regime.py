"""
Market Regime Detector.

Classifies the current market regime based on ADX, ATR percentile,
and Bollinger Band width.  All indicator calculations are self-contained
(no imports from technical_indicators) to avoid circular dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np


class MarketRegime(str, Enum):
    """Discrete market regime labels."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    SQUEEZE = "squeeze"


class MarketRegimeDetector:
    """Detect the prevailing market regime from OHLC data.

    Parameters
    ----------
    adx_period : int
        Look-back period for ADX / DI calculation (default 14).
    atr_period : int
        Look-back period for ATR calculation (default 14).
    bb_period : int
        Look-back period for Bollinger Band width (default 20).
    bb_std : float
        Standard-deviation multiplier for Bollinger Bands (default 2.0).
    """

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ) -> None:
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def detect(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> Dict[str, object]:
        """Return a dict describing the current market regime.

        Parameters
        ----------
        closes : np.ndarray
            Array of close prices (oldest first).
        highs : np.ndarray
            Array of high prices (oldest first).
        lows : np.ndarray
            Array of low prices (oldest first).

        Returns
        -------
        dict
            Keys: ``regime`` (:class:`MarketRegime`), ``adx_value``,
            ``atr_pct``, ``bb_width``, ``squeeze`` (bool).
        """
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)

        adx_value, di_plus, di_minus = self._adx(closes, highs, lows)
        atr_pct = self._atr_percentile(closes, highs, lows)
        bb_width = self._bollinger_width(closes)
        squeeze = bb_width < 0.03

        regime = self._classify(adx_value, di_plus, di_minus, atr_pct, bb_width)

        return {
            "regime": regime,
            "adx_value": round(adx_value, 4),
            "atr_pct": round(atr_pct, 4),
            "bb_width": round(bb_width, 6),
            "squeeze": squeeze,
        }

    # ------------------------------------------------------------------
    # regime classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(
        adx: float,
        di_plus: float,
        di_minus: float,
        atr_pct: float,
        bb_width: float,
    ) -> MarketRegime:
        """Determine the regime from indicator values."""
        if bb_width < 0.03:
            return MarketRegime.SQUEEZE
        if atr_pct > 80.0:
            return MarketRegime.HIGH_VOLATILITY
        if adx > 25.0:
            if di_plus > di_minus:
                return MarketRegime.TRENDING_UP
            return MarketRegime.TRENDING_DOWN
        if adx < 20.0:
            return MarketRegime.RANGING
        # ADX between 20 and 25 -- ambiguous; default to RANGING
        return MarketRegime.RANGING

    # ------------------------------------------------------------------
    # self-contained indicator helpers
    # ------------------------------------------------------------------

    def _adx(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> tuple:
        """Compute ADX, +DI, -DI using Wilder smoothing.

        Returns the most recent (ADX, +DI, -DI) values.
        """
        period = self.adx_period
        n = len(closes)
        if n < period + 1:
            return 0.0, 0.0, 0.0

        # True Range, +DM, -DM
        tr_arr = np.empty(n - 1)
        plus_dm = np.empty(n - 1)
        minus_dm = np.empty(n - 1)

        for i in range(1, n):
            hi = highs[i]
            lo = lows[i]
            prev_close = closes[i - 1]
            tr_arr[i - 1] = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))

            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm[i - 1] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i - 1] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder smoothing (first value = sum of first `period` elements)
        def _wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
            out = np.empty(len(arr) - p + 1)
            out[0] = arr[:p].sum()
            for j in range(1, len(out)):
                out[j] = out[j - 1] - out[j - 1] / p + arr[p + j - 1]
            return out

        atr_smooth = _wilder_smooth(tr_arr, period)
        plus_dm_smooth = _wilder_smooth(plus_dm, period)
        minus_dm_smooth = _wilder_smooth(minus_dm, period)

        length = min(len(atr_smooth), len(plus_dm_smooth), len(minus_dm_smooth))
        if length == 0:
            return 0.0, 0.0, 0.0

        di_plus_arr = 100.0 * plus_dm_smooth[:length] / np.maximum(atr_smooth[:length], 1e-10)
        di_minus_arr = 100.0 * minus_dm_smooth[:length] / np.maximum(atr_smooth[:length], 1e-10)

        dx = 100.0 * np.abs(di_plus_arr - di_minus_arr) / np.maximum(di_plus_arr + di_minus_arr, 1e-10)

        if len(dx) < period:
            adx_val = float(dx[-1]) if len(dx) > 0 else 0.0
        else:
            adx_val = float(dx[:period].mean())
            for k in range(period, len(dx)):
                adx_val = (adx_val * (period - 1) + dx[k]) / period

        return adx_val, float(di_plus_arr[-1]), float(di_minus_arr[-1])

    def _atr_percentile(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> float:
        """Return the percentile rank (0-100) of the latest ATR value
        relative to its own rolling history."""
        period = self.atr_period
        n = len(closes)
        if n < period + 1:
            return 50.0

        tr_values = np.empty(n - 1)
        for i in range(1, n):
            hi = highs[i]
            lo = lows[i]
            prev_close = closes[i - 1]
            tr_values[i - 1] = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))

        # Simple rolling ATR (SMA of TR)
        if len(tr_values) < period:
            return 50.0

        atr_series = np.convolve(tr_values, np.ones(period) / period, mode="valid")
        if len(atr_series) == 0:
            return 50.0

        latest_atr = atr_series[-1]
        pct = float(np.sum(atr_series <= latest_atr) / len(atr_series) * 100.0)
        return pct

    def _bollinger_width(self, closes: np.ndarray) -> float:
        """Return normalised Bollinger Band width (band / middle)."""
        period = self.bb_period
        if len(closes) < period:
            return 0.05  # neutral default

        window = closes[-period:]
        middle = float(np.mean(window))
        if middle == 0.0:
            return 0.05
        std = float(np.std(window, ddof=0))
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        width = (upper - lower) / middle
        return width
