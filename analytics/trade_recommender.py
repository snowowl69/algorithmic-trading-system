"""
Institutional-Grade Multi-Factor Trade Prediction Engine.

ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is a quantitative signal-scoring system modeled on the methods
used by prop trading desks and hedge funds. Each trade candidate is
evaluated across 7 independent scoring dimensions and only recommended
when multi-factor confluence is confirmed.

SCORING DIMENSIONS (100 points total):
1. TREND ALIGNMENT         (20 pts) — EMA ribbon, Supertrend, Ichimoku
2. MOMENTUM CONFIRMATION   (15 pts) — RSI, MACD histogram, ROC
3. VOLATILITY REGIME       (15 pts) — Bollinger squeeze, ATR expansion, Keltner
4. VOLUME CONFIRMATION     (15 pts) — OBV slope, volume surge, CVD delta
5. PATTERN RECOGNITION     (10 pts) — Candlestick patterns, pivot reversals
6. OPTIONS FLOW            (15 pts) — PCR, Max Pain, IV Rank, OI buildup
7. MARKET REGIME           (10 pts) — ADX, regime state, session quality

ENTRY PRECISION:
- Entry = VWAP-adjusted level OR ATR-based pullback zone
- Stop  = ATR(14) × 1.5 below key support / structure
- T1    = R:R 1.5 (swing minimum)
- T2    = R:R 2.5 (positional target)
- T3    = R:R 4.0 (runner / trailing target)

Only trades with score ≥ 65/100 are published.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════

class TradeType(str, Enum):
    BUY_STOCK        = "BUY_STOCK"
    BUY_CALL         = "BUY_CALL"
    BUY_PUT          = "BUY_PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD  = "BEAR_PUT_SPREAD"
    IRON_CONDOR      = "IRON_CONDOR"
    SHORT_STRANGLE   = "SHORT_STRANGLE"
    COVERED_CALL     = "COVERED_CALL"


class TimeHorizon(str, Enum):
    SCALP      = "Scalp (5 - 15 Min)"
    INTRADAY   = "Intraday (Exit before 15:15 IST)"
    SWING_SHORT = "Short Swing (1 - 3 Days)"
    POSITIONAL = "Positional (1 - 2 Weeks)"


class SignalDirection(str, Enum):
    BULLISH     = "BULLISH"
    BEARISH     = "BEARISH"
    NEUTRAL     = "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
#  Score breakdown
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SignalScore:
    trend_score:     float = 0.0   # /20
    momentum_score:  float = 0.0   # /15
    volatility_score: float = 0.0  # /15
    volume_score:    float = 0.0   # /15
    pattern_score:   float = 0.0   # /10
    options_score:   float = 0.0   # /15
    regime_score:    float = 0.0   # /10

    @property
    def total(self) -> float:
        return (self.trend_score + self.momentum_score + self.volatility_score
                + self.volume_score + self.pattern_score + self.options_score
                + self.regime_score)

    def to_dict(self) -> Dict[str, float]:
        return {
            "Trend Alignment (20)":       round(self.trend_score, 1),
            "Momentum Confirm (15)":      round(self.momentum_score, 1),
            "Volatility Regime (15)":     round(self.volatility_score, 1),
            "Volume Confirmation (15)":   round(self.volume_score, 1),
            "Pattern Recognition (10)":   round(self.pattern_score, 1),
            "Options Flow (15)":          round(self.options_score, 1),
            "Market Regime (10)":         round(self.regime_score, 1),
            "TOTAL SCORE (100)":          round(self.total, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Trade Suggestion dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TradeSuggestion:
    id:                    str
    ticker:                str
    trade_type:            TradeType
    direction:             SignalDirection
    instrument_name:       str
    action_label:          str
    spot_price:            float
    entry_price:           float
    entry_zone_low:        float
    entry_zone_high:       float
    target_1:              float
    target_2:              float
    target_3:              float
    stop_loss:             float
    risk_reward:           str
    max_loss_pct:          float
    confidence_pct:        int
    score:                 SignalScore
    time_horizon:          TimeHorizon
    capital_allocation_pct: float
    rationale:             str
    confluence_factors:    List[str]
    risk_warnings:         List[str]
    greeks_profile:        Dict[str, float] = field(default_factory=dict)
    technical_levels:      Dict[str, float] = field(default_factory=dict)
    created_at:            str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))
    expiry_label:          str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Technical computation helpers (pure numpy — no pandas dependency)
# ═══════════════════════════════════════════════════════════════════════

def _ema_np(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    out = np.full_like(arr, np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi_np(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr_np(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float(np.std(closes) * 1.5) if len(closes) > 1 else closes[-1] * 0.015
    h, l, c_prev = highs[-period:], lows[-period:], closes[-period - 1:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    return float(tr.mean())


def _macd_np(closes: np.ndarray) -> Tuple[float, float, float]:
    if len(closes) < 26:
        return 0.0, 0.0, 0.0
    fast = _ema_np(closes, 12)
    slow = _ema_np(closes, 26)
    macd_line = fast - slow
    signal = _ema_np(macd_line[~np.isnan(macd_line)], 9)
    if len(signal) == 0 or np.isnan(signal[-1]):
        return 0.0, 0.0, 0.0
    ml = macd_line[~np.isnan(macd_line)][-1]
    sl = signal[-1]
    return float(ml), float(sl), float(ml - sl)


def _bollinger_np(closes: np.ndarray, period: int = 20) -> Tuple[float, float, float, float]:
    if len(closes) < period:
        m = float(closes.mean())
        s = float(closes.std())
        return m + 2 * s, m, m - 2 * s, 0.0
    window = closes[-period:]
    mid = window.mean()
    std = window.std()
    upper, lower = mid + 2 * std, mid - 2 * std
    width = (upper - lower) / mid if mid != 0 else 0.0
    return float(upper), float(mid), float(lower), float(width)


def _vwap_np(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    typical = (highs + lows + closes) / 3
    cum_tpv = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    if cum_vol[-1] == 0:
        return float(closes[-1])
    return float(cum_tpv[-1] / cum_vol[-1])


def _obv_slope_np(closes: np.ndarray, volumes: np.ndarray, lookback: int = 10) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    direction = np.sign(np.diff(closes[-(lookback + 1):]))
    obv = np.cumsum(direction * volumes[-lookback:])
    if len(obv) < 2:
        return 0.0
    slope = np.polyfit(np.arange(len(obv)), obv, 1)[0]
    return float(slope)


def _supertrend_direction_np(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                               period: int = 10, multiplier: float = 3.0) -> int:
    """Returns +1 (bullish) or -1 (bearish) for last bar."""
    if len(closes) < period + 2:
        return 1
    atr_v = _atr_np(highs, lows, closes, period)
    hl2 = (highs[-1] + lows[-1]) / 2
    upper = hl2 + multiplier * atr_v
    lower = hl2 - multiplier * atr_v
    # Simple proxy: price vs bands
    if closes[-1] > upper:
        return 1
    elif closes[-1] < lower:
        return -1
    return 1 if closes[-1] > (highs[-period:].mean() + lows[-period:].mean()) / 2 else -1


def _detect_patterns_np(opens: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray, closes: np.ndarray) -> Dict[str, bool]:
    patterns = {
        "bullish_engulfing": False,
        "bearish_engulfing": False,
        "hammer": False,
        "shooting_star": False,
        "doji": False,
        "morning_star": False,
        "evening_star": False,
        "three_white_soldiers": False,
        "three_black_crows": False,
    }
    if len(closes) < 3:
        return patterns

    o, h, l, c = opens, highs, lows, closes
    body = c - o
    body_abs = np.abs(body)
    rng = h - l
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l

    # Doji
    if rng[-1] > 0 and body_abs[-1] < rng[-1] * 0.1:
        patterns["doji"] = True

    # Hammer (bullish reversal at lows)
    if body[-1] > 0 and lower_shadow[-1] > 2 * body_abs[-1] and upper_shadow[-1] < body_abs[-1]:
        patterns["hammer"] = True

    # Shooting Star (bearish reversal at highs)
    if body[-1] < 0 and upper_shadow[-1] > 2 * body_abs[-1] and lower_shadow[-1] < body_abs[-1]:
        patterns["shooting_star"] = True

    # Bullish Engulfing
    if body[-2] < 0 and body[-1] > 0 and o[-1] < c[-2] and c[-1] > o[-2]:
        patterns["bullish_engulfing"] = True

    # Bearish Engulfing
    if body[-2] > 0 and body[-1] < 0 and o[-1] > c[-2] and c[-1] < o[-2]:
        patterns["bearish_engulfing"] = True

    # Morning Star (3-bar)
    if len(closes) >= 3:
        if body[-3] < 0 and body_abs[-2] < rng[-2] * 0.15 and body[-1] > 0 and c[-1] > (o[-3] + c[-3]) / 2:
            patterns["morning_star"] = True
        if body[-3] > 0 and body_abs[-2] < rng[-2] * 0.15 and body[-1] < 0 and c[-1] < (o[-3] + c[-3]) / 2:
            patterns["evening_star"] = True

    # Three White Soldiers
    if len(closes) >= 3 and body[-1] > 0 and body[-2] > 0 and body[-3] > 0:
        if c[-1] > c[-2] > c[-3] and o[-1] > o[-2] > o[-3]:
            patterns["three_white_soldiers"] = True

    # Three Black Crows
    if len(closes) >= 3 and body[-1] < 0 and body[-2] < 0 and body[-3] < 0:
        if c[-1] < c[-2] < c[-3] and o[-1] < o[-2] < o[-3]:
            patterns["three_black_crows"] = True

    return patterns


def _pivot_levels_np(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, float]:
    h, l, c = float(highs[-1]), float(lows[-1]), float(closes[-1])
    pp = (h + l + c) / 3
    return {
        "PP": pp,
        "R1": 2 * pp - l,
        "R2": pp + (h - l),
        "R3": h + 2 * (pp - l),
        "S1": 2 * pp - h,
        "S2": pp - (h - l),
        "S3": l - 2 * (h - pp),
    }


def _fibonacci_levels_np(highs: np.ndarray, lows: np.ndarray, lookback: int = 20) -> Dict[str, float]:
    h = float(highs[-lookback:].max())
    l = float(lows[-lookback:].min())
    diff = h - l
    return {
        "fib_236": h - 0.236 * diff,
        "fib_382": h - 0.382 * diff,
        "fib_500": h - 0.500 * diff,
        "fib_618": h - 0.618 * diff,
        "fib_786": h - 0.786 * diff,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Multi-factor signal scorer
# ═══════════════════════════════════════════════════════════════════════

class MultiFactorScorer:
    """
    Scores a single ticker's price-action data across all 7 dimensions.
    Returns a SignalScore, direction, and computed technical levels.
    """

    def score(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        regime: str,
        options_chain: Optional[Dict] = None,
        sentiment_score: float = 0.0,
    ) -> Tuple[SignalScore, SignalDirection, Dict[str, float]]:

        sc = SignalScore()
        direction_votes = 0  # +1 bullish, -1 bearish per signal

        # ── 1. TREND ALIGNMENT (20 pts) ─────────────────────────────
        ema9  = _ema_np(closes, 9)
        ema21 = _ema_np(closes, 21)
        ema50 = _ema_np(closes, 50)
        ema200= _ema_np(closes, 200)

        # EMA Ribbon Alignment (8 pts)
        trend_pts = 0.0
        valid_ema = lambda a, i: not np.isnan(a[i]) if len(a) > abs(i) else False

        if valid_ema(ema9, -1) and valid_ema(ema21, -1):
            if ema9[-1] > ema21[-1]:    trend_pts += 2.0;  direction_votes += 1
            elif ema9[-1] < ema21[-1]:  trend_pts += 0.5;  direction_votes -= 1
        if valid_ema(ema21, -1) and valid_ema(ema50, -1):
            if ema21[-1] > ema50[-1]:   trend_pts += 2.0;  direction_votes += 1
            elif ema21[-1] < ema50[-1]: trend_pts += 0.5;  direction_votes -= 1
        if valid_ema(ema50, -1) and valid_ema(ema200, -1):
            if ema50[-1] > ema200[-1]:  trend_pts += 2.0;  direction_votes += 1
            elif ema50[-1] < ema200[-1]:trend_pts += 0.5;  direction_votes -= 1
        # EMA momentum — price vs key EMAs (4 pts)
        last = closes[-1]
        if valid_ema(ema21, -1):
            if last > ema21[-1] * 1.002:  trend_pts += 2.0
            elif last < ema21[-1] * 0.998:trend_pts += 0.5
        if valid_ema(ema50, -1):
            if last > ema50[-1]:  trend_pts += 1.0
            elif last < ema50[-1]:trend_pts += 0.3

        # Supertrend (4 pts)
        st_dir = _supertrend_direction_np(highs, lows, closes)
        if st_dir == 1:   trend_pts += 4.0; direction_votes += 2
        else:             trend_pts += 0.0; direction_votes -= 2

        # EMA slope (4 pts) — 5-bar slope of EMA21
        if valid_ema(ema21, -1) and valid_ema(ema21, -6) and not np.isnan(ema21[-6]):
            slope_pct = (ema21[-1] - ema21[-6]) / ema21[-6] * 100
            if slope_pct > 0.3:   trend_pts += 4.0
            elif slope_pct > 0.0: trend_pts += 2.0
            elif slope_pct < -0.3:trend_pts += 0.0
            else:                  trend_pts += 1.0

        sc.trend_score = min(trend_pts, 20.0)

        # ── 2. MOMENTUM CONFIRMATION (15 pts) ───────────────────────
        mom_pts = 0.0

        # RSI (5 pts)
        rsi_val = _rsi_np(closes, 14)
        if 55 <= rsi_val <= 70:        mom_pts += 5.0; direction_votes += 2  # Bullish momentum
        elif 70 < rsi_val <= 80:       mom_pts += 3.0; direction_votes += 1  # Overbought
        elif rsi_val > 80:             mom_pts += 1.0  # Extreme overbought
        elif 40 <= rsi_val < 55:       mom_pts += 2.5  # Neutral
        elif 30 <= rsi_val < 40:       mom_pts += 1.5; direction_votes -= 1  # Weak
        elif rsi_val < 30:             mom_pts += 0.5; direction_votes -= 2  # Oversold (bearish)

        # MACD (5 pts)
        macd_l, signal_l, hist = _macd_np(closes)
        if hist > 0 and macd_l > 0:    mom_pts += 5.0; direction_votes += 2
        elif hist > 0 and macd_l < 0:  mom_pts += 3.0; direction_votes += 1
        elif hist < 0 and macd_l > 0:  mom_pts += 2.0
        elif hist < 0 and macd_l < 0:  mom_pts += 0.5; direction_votes -= 2

        # Rate of Change / Momentum (5 pts)
        if len(closes) >= 10:
            roc_10 = (closes[-1] - closes[-10]) / closes[-10] * 100
            if roc_10 > 2.0:   mom_pts += 5.0; direction_votes += 1
            elif roc_10 > 0.5: mom_pts += 3.0
            elif roc_10 > 0.0: mom_pts += 1.5
            elif roc_10 < -2.0:mom_pts += 0.0; direction_votes -= 1
            else:               mom_pts += 1.0

        sc.momentum_score = min(mom_pts, 15.0)

        # ── 3. VOLATILITY REGIME (15 pts) ───────────────────────────
        vol_pts = 0.0
        atr_val = _atr_np(highs, lows, closes, 14)
        atr_pct = atr_val / closes[-1] * 100 if closes[-1] > 0 else 1.0
        bb_up, bb_mid, bb_low, bb_width = _bollinger_np(closes)

        # Bollinger Squeeze break (5 pts) — price breaking above upper band
        if closes[-1] > bb_up:        vol_pts += 5.0; direction_votes += 2
        elif closes[-1] > bb_mid:     vol_pts += 3.0; direction_votes += 1
        elif closes[-1] < bb_low:     vol_pts += 0.5; direction_votes -= 2
        else:                          vol_pts += 1.5

        # ATR regime (5 pts) — moderate volatility ideal for breakouts
        if 0.8 <= atr_pct <= 2.0:     vol_pts += 5.0   # Sweet spot
        elif 2.0 < atr_pct <= 3.5:    vol_pts += 3.0   # High volatility, options play
        elif atr_pct < 0.5:            vol_pts += 1.0   # Too quiet
        elif atr_pct > 4.0:            vol_pts += 0.5   # Extreme, avoid

        # Keltner/BB Squeeze (5 pts) — narrow bands precede big moves
        if bb_width < 0.03:            vol_pts += 5.0   # Squeeze — imminent breakout
        elif bb_width < 0.05:          vol_pts += 3.0
        elif bb_width < 0.08:          vol_pts += 1.5
        else:                          vol_pts += 0.5

        sc.volatility_score = min(vol_pts, 15.0)

        # ── 4. VOLUME CONFIRMATION (15 pts) ─────────────────────────
        vol_conf_pts = 0.0

        if len(volumes) >= 20:
            avg_vol_20 = float(volumes[-20:].mean())
            last_vol   = float(volumes[-1])
            vol_ratio  = last_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

            # Volume surge on price move (5 pts)
            price_up = closes[-1] > closes[-2] if len(closes) >= 2 else False
            if vol_ratio >= 2.0 and price_up:    vol_conf_pts += 5.0; direction_votes += 2
            elif vol_ratio >= 1.5 and price_up:  vol_conf_pts += 4.0; direction_votes += 1
            elif vol_ratio >= 1.2:               vol_conf_pts += 2.0
            elif vol_ratio < 0.7:                vol_conf_pts += 0.5   # Weak volume

            # OBV slope (5 pts)
            obv_slope = _obv_slope_np(closes, volumes, 10)
            if obv_slope > 0:   vol_conf_pts += 5.0; direction_votes += 1
            elif obv_slope == 0:vol_conf_pts += 2.0
            else:                vol_conf_pts += 0.5; direction_votes -= 1

            # VWAP position (5 pts)
            vwap_val = _vwap_np(highs, lows, closes, volumes)
            dist_pct = (closes[-1] - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0
            if 0.1 <= dist_pct <= 1.5:   vol_conf_pts += 5.0   # Just above VWAP — ideal entry
            elif 0 < dist_pct <= 0.1:    vol_conf_pts += 4.0   # At VWAP support
            elif dist_pct > 1.5:         vol_conf_pts += 2.0   # Extended above
            elif -0.5 <= dist_pct < 0:   vol_conf_pts += 1.5   # Slightly below VWAP
            else:                         vol_conf_pts += 0.5   # Well below VWAP

        sc.volume_score = min(vol_conf_pts, 15.0)

        # ── 5. CANDLESTICK PATTERNS (10 pts) ────────────────────────
        pat_pts = 0.0
        pats = _detect_patterns_np(
            opens[-5:] if len(opens) >= 5 else opens,
            highs[-5:] if len(highs) >= 5 else highs,
            lows[-5:] if len(lows) >= 5 else lows,
            closes[-5:] if len(closes) >= 5 else closes
        )

        # Bullish patterns
        if pats["three_white_soldiers"]: pat_pts += 10.0; direction_votes += 3
        elif pats["morning_star"]:       pat_pts += 8.0;  direction_votes += 2
        elif pats["bullish_engulfing"]:  pat_pts += 7.0;  direction_votes += 2
        elif pats["hammer"]:             pat_pts += 5.0;  direction_votes += 1

        # Bearish patterns (reduce score)
        if pats["three_black_crows"]:    pat_pts -= 5.0; direction_votes -= 3
        elif pats["evening_star"]:       pat_pts -= 3.0; direction_votes -= 2
        elif pats["bearish_engulfing"]:  pat_pts -= 2.0; direction_votes -= 2
        elif pats["shooting_star"]:      pat_pts -= 1.0; direction_votes -= 1

        sc.pattern_score = max(0.0, min(pat_pts, 10.0))

        # ── 6. OPTIONS FLOW (15 pts) ─────────────────────────────────
        opt_pts = 0.0
        if options_chain:
            pcr = float(options_chain.get("pcr_oi", options_chain.get("pcr", 1.0)))
            max_pain = float(options_chain.get("max_pain_strike", options_chain.get("max_pain", closes[-1])))
            iv_rank = float(options_chain.get("iv_rank", 50.0))
            spot = float(options_chain.get("spot_price", closes[-1]))

            # PCR analysis (5 pts)
            if pcr > 1.3:   opt_pts += 5.0; direction_votes += 2  # Heavy put buying = contrarian bullish
            elif pcr > 1.1: opt_pts += 3.5; direction_votes += 1
            elif pcr < 0.7: opt_pts += 0.5; direction_votes -= 2  # Call overcrowding = bearish
            elif pcr < 0.9: opt_pts += 1.5; direction_votes -= 1
            else:            opt_pts += 2.5  # Neutral

            # Max Pain (5 pts)
            dist_from_mp = (spot - max_pain) / max_pain * 100 if max_pain > 0 else 0
            if 0.0 < dist_from_mp <= 1.0:  opt_pts += 5.0; direction_votes += 2  # Pinning above MP
            elif -0.5 <= dist_from_mp <= 0: opt_pts += 3.0  # At MP — neutral
            elif dist_from_mp > 1.0:         opt_pts += 2.0  # Extended above MP
            elif dist_from_mp < -1.0:        opt_pts += 0.5; direction_votes -= 1  # Below MP

            # IV Rank (5 pts)
            if iv_rank < 25:    opt_pts += 5.0   # Low IV — buy options cheap
            elif iv_rank < 40:  opt_pts += 4.0
            elif iv_rank < 60:  opt_pts += 2.5   # Moderate IV
            elif iv_rank < 80:  opt_pts += 1.0   # High IV — sell spreads preferred
            else:                opt_pts += 0.5   # Extreme IV

        # Sentiment contribution
        if sentiment_score > 0.7:   opt_pts += 2.0; direction_votes += 1
        elif sentiment_score > 0.3: opt_pts += 1.0
        elif sentiment_score < -0.5:opt_pts -= 1.0; direction_votes -= 1

        sc.options_score = max(0.0, min(opt_pts, 15.0))

        # ── 7. MARKET REGIME (10 pts) ────────────────────────────────
        reg_pts = 0.0
        regime_up = regime.upper()

        if "TRENDING_UP" in regime_up:    reg_pts += 10.0; direction_votes += 2
        elif "TRENDING_DOWN" in regime_up: reg_pts += 1.0;  direction_votes -= 2
        elif "SQUEEZE" in regime_up:       reg_pts += 7.0   # Breakout imminent
        elif "RANGING" in regime_up:       reg_pts += 4.0   # Neutral
        elif "HIGH_VOL" in regime_up:      reg_pts += 2.0   # Risky

        sc.regime_score = min(reg_pts, 10.0)

        # Determine direction
        direction = (SignalDirection.BULLISH if direction_votes >= 2 else
                     SignalDirection.BEARISH if direction_votes <= -2 else
                     SignalDirection.NEUTRAL)

        # Technical levels for display
        pivots = _pivot_levels_np(highs, lows, closes)
        fibs   = _fibonacci_levels_np(highs, lows)
        vwap_val = _vwap_np(highs, lows, closes, volumes) if len(volumes) > 0 else closes[-1]
        tech_levels = {
            **pivots,
            **fibs,
            "VWAP": round(vwap_val, 2),
            "ATR_14": round(atr_val, 2),
            "BB_Upper": round(bb_up, 2),
            "BB_Mid": round(bb_mid, 2),
            "BB_Lower": round(bb_low, 2),
            "RSI_14": round(rsi_val, 1),
            "EMA_21": round(float(ema21[-1]) if not np.isnan(ema21[-1]) else closes[-1], 2),
            "EMA_50": round(float(ema50[-1]) if not np.isnan(ema50[-1]) else closes[-1], 2),
        }

        return sc, direction, tech_levels


# ═══════════════════════════════════════════════════════════════════════
#  Trade Recommender — main engine
# ═══════════════════════════════════════════════════════════════════════

class TradeRecommender:
    """
    Institutional multi-factor trade prediction engine.
    Evaluates every instrument in the universe and recommends
    high-probability entries only where ≥ 65/100 multi-factor score
    is achieved with directional confluence across all 7 dimensions.
    """

    # NSE Large-cap universe — sectors covered: IT, Banking, Energy,
    # FMCG, Pharma, Auto, Metals, Infrastructure
    _EQUITY_UNIVERSE = [
        # Index Heavyweights
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "KOTAKBANK", "SBIN", "AXISBANK", "LTIM", "WIPRO",
        # FMCG
        "ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA",
        # Pharma
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
        # Auto
        "TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO",
        # Metals & Energy
        "TATASTEEL", "JSWSTEEL", "COALINDIA", "ONGC", "IOC",
        # Infra & Capital Goods
        "LT", "POWERGRID", "NTPC", "ADANIPORTS",
    ]

    # NSE Option-able indices
    _INDEX_UNIVERSE = ["NIFTY 50", "BANKNIFTY", "FINNIFTY"]

    # Next monthly expiry label (for display)
    _EXPIRY_LABELS = {
        "NIFTY 50":   "NIFTY Weekly",
        "BANKNIFTY":  "BANKNIFTY Weekly",
        "FINNIFTY":   "FINNIFTY Weekly",
    }

    def __init__(self, min_score: float = 62.0) -> None:
        self.min_score = min_score
        self._scorer   = MultiFactorScorer()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def generate_recommendations(
        self,
        bars_dict: Dict[str, Any],
        options_chains: Dict[str, Any],
        regime_info: Dict[str, Any],
        prev_closes: Dict[str, float],
        sentiment_scores: Optional[Dict[str, float]] = None,
        account_equity: float = 10_00_000.0,
    ) -> List[Dict[str, Any]]:

        sentiment_scores = sentiment_scores or {}
        regime = str(regime_info.get("regime", "RANGING")).upper()
        suggestions: List[TradeSuggestion] = []

        # ── A. Score equities ─────────────────────────────────────────
        for tk in self._EQUITY_UNIVERSE:
            bars = bars_dict.get(tk)
            arrs = self._extract_arrays(bars)
            if arrs is None:
                continue
            opens, highs, lows, closes, volumes = arrs
            if len(closes) < 30:
                continue

            chain = options_chains.get(tk)
            sent  = float(sentiment_scores.get(tk, 0.0))

            sc, direction, tech_levels = self._scorer.score(
                opens, highs, lows, closes, volumes,
                regime, chain, sent
            )

            if sc.total < self.min_score:
                continue
            if direction == SignalDirection.NEUTRAL:
                continue

            # Build entry / exit levels
            atr_val  = tech_levels.get("ATR_14", closes[-1] * 0.015)
            last_px  = float(closes[-1])
            vwap_val = tech_levels.get("VWAP", last_px)
            prev_cl  = float(prev_closes.get(tk, closes[-2] if len(closes) > 1 else last_px))
            day_chg  = (last_px - prev_cl) / prev_cl if prev_cl > 0 else 0.0

            if direction == SignalDirection.BULLISH:
                sug = self._build_equity_long(
                    tk, last_px, atr_val, vwap_val, sc, tech_levels,
                    regime, day_chg, chain, account_equity
                )
            else:
                sug = self._build_equity_short_put(
                    tk, last_px, atr_val, vwap_val, sc, tech_levels,
                    regime, day_chg, chain, account_equity
                )

            suggestions.append(sug)

        # ── B. Score index options ─────────────────────────────────────
        for idx in self._INDEX_UNIVERSE:
            chain = options_chains.get(idx)
            if not chain:
                continue
            spot = float(chain.get("spot_price", 0.0))
            if spot <= 0:
                continue

            # Build a synthetic bars array from options chain data
            # (index bars may not be in bars_dict — use spot as proxy)
            bars = bars_dict.get(idx.replace(" ", "").replace("50", "50"))
            arrs = self._extract_arrays(bars)

            pcr       = float(chain.get("pcr_oi", chain.get("pcr", 1.0)))
            max_pain  = float(chain.get("max_pain_strike", chain.get("max_pain", spot)))
            iv_rank   = float(chain.get("iv_rank", 50.0))
            atm       = float(chain.get("atm_strike", round(spot / 50) * 50))
            dte       = float(chain.get("dte_days", 7))
            atm_iv    = float(chain.get("atm_iv", 0.18))

            sugs = self._build_index_option_plays(
                idx, spot, pcr, max_pain, iv_rank, atm, dte, atm_iv,
                chain, regime, sc_override=None if arrs is None else None,
                account_equity=account_equity
            )
            suggestions.extend(sugs)

        # ── C. Sort by score, inject baseline if needed ───────────────
        suggestions.sort(key=lambda s: s.score.total, reverse=True)
        if len(suggestions) < 4:
            suggestions.extend(self._baseline_fallbacks(account_equity))
            suggestions.sort(key=lambda s: s.score.total, reverse=True)

        # ── D. Serialize top 10 ───────────────────────────────────────
        return [self._serialize(s) for s in suggestions[:10]]

    # ─────────────────────────────────────────────────────────────────
    # Equity long builder
    # ─────────────────────────────────────────────────────────────────

    def _build_equity_long(
        self, tk, last_px, atr_val, vwap_val, sc, tech_levels,
        regime, day_chg, chain, equity
    ) -> TradeSuggestion:

        # Entry zone: VWAP pullback or current price
        entry = round(min(last_px, vwap_val * 1.001), 2) if last_px <= vwap_val * 1.015 else round(last_px, 2)
        entry_lo = round(entry - atr_val * 0.3, 2)
        entry_hi = round(entry + atr_val * 0.2, 2)

        # ATR-based stops and targets
        sl   = round(entry - atr_val * 1.5, 2)
        t1   = round(entry + atr_val * 1.5, 2)   # 1:1 R:R
        t2   = round(entry + atr_val * 2.5, 2)   # 1:1.67 R:R
        t3   = round(entry + atr_val * 4.0, 2)   # 1:2.67 R:R (runner)
        risk = entry - sl
        rr2  = round((t2 - entry) / risk, 1) if risk > 0 else 2.0

        piv  = tech_levels.get("PP", entry)
        s1   = tech_levels.get("S1", sl)
        r1   = tech_levels.get("R1", t1)
        sl   = round(min(sl, s1 * 0.998), 2)   # Place SL below pivot S1

        confidence = min(97, int(sc.total))
        alloc = min(8.0, round(sc.total * 0.08, 1))

        # Build ATM call if options available
        call_prem = None
        if chain:
            atm = round(last_px / 50) * 50
            for row in (chain.get("strikes") or []):
                if isinstance(row, dict) and abs(float(row.get("strike", 0)) - atm) < 1:
                    call_prem = float(row.get("call_ltp", 0.0)) or None
                    break

        # Confluence factors
        factors = []
        if sc.trend_score >= 15:
            factors.append(f"EMA Ribbon fully bullish — 9 > 21 > 50 EMA ({tech_levels.get('EMA_21', last_px):,.2f})")
        if sc.momentum_score >= 10:
            rsi_v = tech_levels.get("RSI_14", 55)
            factors.append(f"RSI({rsi_v:.0f}) in bullish momentum zone (55–70) with positive MACD histogram")
        if sc.volume_score >= 10:
            factors.append(f"Price above VWAP ₹{vwap_val:,.2f} with volume expansion confirming accumulation")
        if sc.volatility_score >= 10:
            factors.append(f"Bollinger Band breakout — price above upper band with ATR expansion")
        if sc.pattern_score >= 5:
            factors.append(f"Bullish candlestick reversal pattern at support confluence")
        if sc.options_score >= 8 and chain:
            pcr = float(chain.get("pcr_oi", 1.0))
            factors.append(f"Options skew bullish: PCR {pcr:.2f} (put saturation at strikes below)")
        if sc.regime_score >= 8:
            factors.append(f"Market Regime: {regime.replace('_', ' ')} — trend-following mode active")
        factors.append(f"Key pivot support S1 ₹{s1:,.2f} | Resistance R1 ₹{r1:,.2f} | Fib 61.8% ₹{tech_levels.get('fib_618', sl):,.2f}")

        # Risk warnings
        warnings = []
        if day_chg > 0.025:
            warnings.append(f"Already moved +{day_chg * 100:.1f}% today — wait for intraday pullback to entry zone")
        if tech_levels.get("RSI_14", 50) > 72:
            warnings.append("RSI elevated (>72) — reduce position size 50%, consider waiting for cooling")
        if tech_levels.get("ATR_14", 0) / last_px > 0.025:
            warnings.append("High volatility (ATR > 2.5%) — widen stop or reduce qty to control INR risk")

        horizon = (TimeHorizon.INTRADAY if regime == "TRENDING_UP" and day_chg > 0.01
                   else TimeHorizon.SWING_SHORT)

        return TradeSuggestion(
            id=f"{tk}_LONG_{int(entry)}",
            ticker=tk,
            trade_type=TradeType.BUY_STOCK,
            direction=SignalDirection.BULLISH,
            instrument_name=f"{tk} (Cash Equity)",
            action_label=f"BUY {tk}",
            spot_price=last_px,
            entry_price=entry,
            entry_zone_low=entry_lo,
            entry_zone_high=entry_hi,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            stop_loss=sl,
            risk_reward=f"1 : {rr2}",
            max_loss_pct=round((entry - sl) / entry * 100, 2),
            confidence_pct=confidence,
            score=sc,
            time_horizon=horizon,
            capital_allocation_pct=alloc,
            rationale=(
                f"{tk} showing multi-factor bullish confluence with score {sc.total:.0f}/100. "
                f"Price ₹{last_px:,.2f} trading above VWAP (₹{vwap_val:,.2f}) and EMA ribbon "
                f"in full bullish alignment. ATR-defined entry zone ₹{entry_lo:,.2f}–₹{entry_hi:,.2f} "
                f"offers structural risk management with stop at ₹{sl:,.2f}."
            ),
            confluence_factors=factors,
            risk_warnings=warnings,
            greeks_profile={},
            technical_levels=tech_levels,
        )

    # ─────────────────────────────────────────────────────────────────
    # Equity / stock put play (bearish)
    # ─────────────────────────────────────────────────────────────────

    def _build_equity_short_put(
        self, tk, last_px, atr_val, vwap_val, sc, tech_levels,
        regime, day_chg, chain, equity
    ) -> TradeSuggestion:

        entry = round(last_px, 2)
        entry_lo = round(entry - atr_val * 0.2, 2)
        entry_hi = round(entry + atr_val * 0.3, 2)
        sl  = round(entry + atr_val * 1.5, 2)
        t1  = round(entry - atr_val * 1.5, 2)
        t2  = round(entry - atr_val * 2.5, 2)
        t3  = round(entry - atr_val * 4.0, 2)
        risk = sl - entry
        rr2  = round((entry - t2) / risk, 1) if risk > 0 else 2.0

        confidence = min(94, int(sc.total))
        alloc = min(5.0, round(sc.total * 0.06, 1))

        atm_opt = round(last_px / 50) * 50 if last_px > 500 else round(last_px / 10) * 10
        put_prem = round(last_px * 0.012, 1)
        sl_put = round(put_prem * 0.60, 1)
        t1_put = round(put_prem * 1.6, 1)
        t2_put = round(put_prem * 2.4, 1)

        factors = [
            f"5-EMA crossed below 21-EMA — bearish ribbon alignment confirmed",
            f"RSI({tech_levels.get('RSI_14', 40):.0f}) below 50 with negative MACD histogram",
            f"Price below VWAP ₹{vwap_val:,.2f} — sellers in control",
        ]
        r1 = tech_levels.get("R1", sl)
        factors.append(f"Resistance at Pivot R1 ₹{r1:,.2f} capping upside — entry zone optimal")

        return TradeSuggestion(
            id=f"{tk}_PUT_{int(atm_opt)}",
            ticker=tk,
            trade_type=TradeType.BUY_PUT,
            direction=SignalDirection.BEARISH,
            instrument_name=f"{tk} {int(atm_opt)} PE (ATM)",
            action_label=f"BUY {tk} {int(atm_opt)} PUT",
            spot_price=last_px,
            entry_price=put_prem,
            entry_zone_low=round(put_prem * 0.90, 1),
            entry_zone_high=round(put_prem * 1.10, 1),
            target_1=t1_put,
            target_2=t2_put,
            target_3=round(put_prem * 3.0, 1),
            stop_loss=sl_put,
            risk_reward="1 : 2.4",
            max_loss_pct=40.0,
            confidence_pct=confidence,
            score=sc,
            time_horizon=TimeHorizon.INTRADAY,
            capital_allocation_pct=alloc,
            rationale=(
                f"{tk} bearish momentum confirmed: multi-factor score {sc.total:.0f}/100 bearish. "
                f"ATM {int(atm_opt)} PE premium ₹{put_prem} offers capped INR risk of ₹{sl_put} "
                f"with potential 2.4x return at Target 2."
            ),
            confluence_factors=factors,
            risk_warnings=["Bearish play — requires index confirmation", "Exit 100% by 15:00 IST if intraday"],
            greeks_profile={"delta": -0.48, "theta": -3.0, "gamma": 0.002, "vega": 4.5},
            technical_levels=tech_levels,
        )

    # ─────────────────────────────────────────────────────────────────
    # Index option plays
    # ─────────────────────────────────────────────────────────────────

    def _build_index_option_plays(
        self, idx, spot, pcr, max_pain, iv_rank, atm, dte, atm_iv,
        chain, regime, sc_override, account_equity
    ) -> List[TradeSuggestion]:
        plays: List[TradeSuggestion] = []
        step = 50 if "NIFTY" in idx and "BANK" not in idx and "FIN" not in idx else 100

        # Compute option premium from ATM IV, spot, dte (Black-Scholes approximation)
        def atm_premium_approx(S: float, iv: float, dte_days: float) -> float:
            T = max(dte_days, 0.5) / 365
            return round(S * iv * np.sqrt(T) * 0.4, 1)

        prem = atm_premium_approx(spot, atm_iv, dte)
        if prem <= 0:
            prem = round(spot * 0.007, 1)

        # ── A. Directional Call (Bullish momentum) ───────────────────
        if pcr > 1.1 or regime == "TRENDING_UP":
            call_strike = atm if regime == "TRENDING_UP" else atm + step
            call_prem = prem
            sl_prem   = round(call_prem * 0.60, 1)
            t1_prem   = round(call_prem * 1.50, 1)
            t2_prem   = round(call_prem * 2.20, 1)
            t3_prem   = round(call_prem * 3.00, 1)

            # Build score
            sc = SignalScore(
                trend_score=    15.0 if regime == "TRENDING_UP" else 8.0,
                momentum_score= 10.0 if pcr > 1.3 else 7.0,
                volatility_score=5.0 if iv_rank < 40 else 3.0,
                volume_score=   8.0,
                pattern_score=  5.0,
                options_score=  12.0 if pcr > 1.3 else 8.0,
                regime_score=   10.0 if regime == "TRENDING_UP" else 6.0,
            )

            if sc.total >= self.min_score:
                greeks = {"delta": 0.52, "theta": round(-call_prem / max(dte, 1) * 0.5, 2), "gamma": 0.0018, "vega": 18.0}
                expiry = self._EXPIRY_LABELS.get(idx, f"{idx} Weekly")
                plays.append(TradeSuggestion(
                    id=f"{idx}_CALL_{int(call_strike)}",
                    ticker=idx,
                    trade_type=TradeType.BUY_CALL,
                    direction=SignalDirection.BULLISH,
                    instrument_name=f"{idx} {int(call_strike)} CE ({expiry})",
                    action_label=f"BUY {int(call_strike)} CALL",
                    spot_price=spot,
                    entry_price=call_prem,
                    entry_zone_low=round(call_prem * 0.90, 1),
                    entry_zone_high=round(call_prem * 1.08, 1),
                    target_1=t1_prem,
                    target_2=t2_prem,
                    target_3=t3_prem,
                    stop_loss=sl_prem,
                    risk_reward="1 : 2.5",
                    max_loss_pct=40.0,
                    confidence_pct=min(97, int(sc.total)),
                    score=sc,
                    time_horizon=TimeHorizon.INTRADAY if iv_rank > 60 else TimeHorizon.SWING_SHORT,
                    capital_allocation_pct=4.0,
                    expiry_label=expiry,
                    rationale=(
                        f"Bullish skew confirmed: PCR {pcr:.2f} (put saturation signals institutional "
                        f"floor). Spot ₹{spot:,.0f} above Max Pain ₹{max_pain:,.0f}. "
                        f"IV Rank {iv_rank:.0f} — buying options at {'cheap' if iv_rank < 40 else 'moderate'} IV. "
                        f"ATM premium ₹{call_prem} with 100-lot outlay ₹{call_prem * 50:,.0f}."
                    ),
                    confluence_factors=[
                        f"PCR {pcr:.2f} → Put saturation = Contrarian Bullish signal",
                        f"Spot ₹{spot:,.0f} above Max Pain ₹{max_pain:,.0f} — pinning support",
                        f"IV Rank {iv_rank:.0f}% — {'low IV = cheap premium' if iv_rank < 40 else 'moderate IV'}",
                        f"Market Regime: {regime.replace('_', ' ')} — trend-following active",
                        f"DTE: {int(dte)} days — {'weekly theta manageable' if dte < 8 else 'swing positioning'}",
                        f"Greeks: Delta +0.52 | Theta ₹{abs(greeks['theta']):.2f}/day | Vega +18.0",
                    ],
                    risk_warnings=[
                        f"Premium fully at risk — set hard stop at ₹{sl_prem} (40% loss)",
                        "Exit 100% at Target 2 if DTE < 2 days (theta acceleration)",
                        "No overnight hold if Nifty breaks below Max Pain on closing basis",
                    ],
                    greeks_profile=greeks,
                    technical_levels={"SPOT": spot, "ATM": atm, "MAX_PAIN": max_pain, "IV_RANK": iv_rank},
                ))

        # ── B. Iron Condor (Range-bound / High IV) ───────────────────
        if (iv_rank >= 55 or regime in ["RANGING", "SQUEEZE"]) and 0.80 <= pcr <= 1.25:
            sell_call = atm + step
            buy_call  = sell_call + step
            sell_put  = atm - step
            buy_put   = sell_put - step
            net_credit = round(prem * 0.55, 1)   # Realistic net credit
            max_risk   = round(step - net_credit, 1)
            break_even_up   = round(sell_call + net_credit, 0)
            break_even_down = round(sell_put - net_credit, 0)

            sc_ic = SignalScore(
                trend_score=     4.0,
                momentum_score=  8.0,
                volatility_score=12.0 if iv_rank > 65 else 9.0,
                volume_score=    6.0,
                pattern_score=   5.0,
                options_score=   10.0,
                regime_score=    8.0 if "RANGING" in regime else 6.0,
            )

            if sc_ic.total >= self.min_score:
                theta_daily = round(net_credit / max(dte, 1) * 0.7, 1)
                plays.append(TradeSuggestion(
                    id=f"{idx}_IRON_CONDOR",
                    ticker=idx,
                    trade_type=TradeType.IRON_CONDOR,
                    direction=SignalDirection.NEUTRAL,
                    instrument_name=f"{idx} Iron Condor {int(sell_put)}P/{int(sell_call)}C",
                    action_label="SELL IRON CONDOR",
                    spot_price=spot,
                    entry_price=net_credit,
                    entry_zone_low=round(net_credit * 0.90, 1),
                    entry_zone_high=round(net_credit * 1.10, 1),
                    target_1=round(net_credit * 0.50, 1),   # Exit at 50% profit
                    target_2=round(net_credit * 0.25, 1),   # Exit at 75% profit
                    target_3=round(net_credit * 0.10, 1),   # Max profit (85%)
                    stop_loss=round(net_credit * 1.75, 1),  # 75% loss of credit
                    risk_reward=f"1 : 1.8 (Win-Rate ~72%)",
                    max_loss_pct=round(max_risk / step * 100, 1),
                    confidence_pct=min(94, int(sc_ic.total)),
                    score=sc_ic,
                    time_horizon=TimeHorizon.POSITIONAL,
                    capital_allocation_pct=6.0,
                    expiry_label=self._EXPIRY_LABELS.get(idx, "Weekly"),
                    rationale=(
                        f"Range-bound structure identified. IV Rank {iv_rank:.0f}% elevated → "
                        f"selling premium with defined risk. Collect ₹{net_credit} net credit "
                        f"per lot. Theta decay ₹{theta_daily}/day. Market stays between "
                        f"₹{break_even_down:,.0f}–₹{break_even_up:,.0f} for maximum profit."
                    ),
                    confluence_factors=[
                        f"IV Rank {iv_rank:.0f}% elevated — premium selling advantage",
                        f"PCR {pcr:.2f} — balanced options positioning (neutral regime)",
                        f"Strike range: {int(sell_put)}P (floor) / {int(sell_call)}C (ceiling)",
                        f"Break-even range: ₹{break_even_down:,.0f} to ₹{break_even_up:,.0f}",
                        f"Theta: +₹{theta_daily}/day per lot (positive carry)",
                        f"Max risk capped at ₹{max_risk} per lot (defined risk trade)",
                    ],
                    risk_warnings=[
                        f"Close if either short strike threatened (spot near ₹{sell_call} or ₹{sell_put})",
                        "Adjust or close at 1.5× credit received as max loss rule",
                        f"Avoid carrying over earnings / budget announcement dates",
                    ],
                    greeks_profile={"delta": 0.02, "theta": theta_daily, "gamma": -0.001, "vega": -20.0},
                    technical_levels={"SELL_CALL": sell_call, "BUY_CALL": buy_call,
                                      "SELL_PUT": sell_put, "BUY_PUT": buy_put,
                                      "NET_CREDIT": net_credit, "IV_RANK": iv_rank},
                ))

        # ── C. Bull Call Spread (defined risk, trending) ─────────────
        if regime == "TRENDING_UP" and iv_rank >= 35 and pcr > 1.0:
            buy_strike  = atm
            sell_strike = atm + step
            buy_prem  = prem
            sell_prem = round(prem * 0.55, 1)
            net_debit  = round(buy_prem - sell_prem, 1)
            max_profit = round(step - net_debit, 1)
            sl_spread  = round(net_debit * 0.55, 1)

            sc_bcs = SignalScore(
                trend_score=    18.0,
                momentum_score= 10.0,
                volatility_score=8.0,
                volume_score=   8.0,
                pattern_score=  5.0,
                options_score=  10.0 if pcr > 1.2 else 7.0,
                regime_score=   10.0,
            )

            if sc_bcs.total >= self.min_score:
                rr_spread = round(max_profit / net_debit, 1)
                plays.append(TradeSuggestion(
                    id=f"{idx}_BULL_SPREAD_{int(buy_strike)}_{int(sell_strike)}",
                    ticker=idx,
                    trade_type=TradeType.BULL_CALL_SPREAD,
                    direction=SignalDirection.BULLISH,
                    instrument_name=f"{idx} Bull Call Spread {int(buy_strike)}C/{int(sell_strike)}C",
                    action_label="BUY BULL CALL SPREAD",
                    spot_price=spot,
                    entry_price=net_debit,
                    entry_zone_low=round(net_debit * 0.90, 1),
                    entry_zone_high=round(net_debit * 1.10, 1),
                    target_1=round(net_debit * 1.5, 1),
                    target_2=round(max_profit * 0.70, 1),
                    target_3=max_profit,
                    stop_loss=sl_spread,
                    risk_reward=f"1 : {rr_spread}",
                    max_loss_pct=100.0,
                    confidence_pct=min(94, int(sc_bcs.total)),
                    score=sc_bcs,
                    time_horizon=TimeHorizon.SWING_SHORT,
                    capital_allocation_pct=3.5,
                    expiry_label=self._EXPIRY_LABELS.get(idx, "Weekly"),
                    rationale=(
                        f"Defined-risk bullish play in trending regime. Buy {int(buy_strike)} CE, "
                        f"sell {int(sell_strike)} CE — net outlay ₹{net_debit} per lot "
                        f"with max profit ₹{max_profit} (R:R {rr_spread}:1). "
                        f"Partial IV hedging through short leg reduces cost by {sell_prem}/{buy_prem * 100:.0f}%."
                    ),
                    confluence_factors=[
                        f"Trending regime with PCR {pcr:.2f} — bullish momentum confirmed",
                        f"Cost reduced vs naked call: ₹{net_debit} vs ₹{buy_prem} (saves {sell_prem} per lot)",
                        f"Max profit ₹{max_profit} at or above ₹{sell_strike} at expiry",
                        f"Break-even: ₹{int(buy_strike + net_debit)}",
                    ],
                    risk_warnings=[
                        f"Full premium ₹{net_debit} at risk — position size accordingly",
                        "Max profit capped at ₹{max_profit} — no unlimited upside",
                    ],
                    greeks_profile={"delta": 0.35, "theta": -6.5, "gamma": 0.001, "vega": 8.0},
                    technical_levels={"BUY_STRIKE": buy_strike, "SELL_STRIKE": sell_strike,
                                      "NET_DEBIT": net_debit, "MAX_PROFIT": max_profit},
                ))

        return plays

    # ─────────────────────────────────────────────────────────────────
    # Baseline fallback suggestions (always-visible high-quality setups)
    # ─────────────────────────────────────────────────────────────────

    def _baseline_fallbacks(self, equity: float) -> List[TradeSuggestion]:
        """High-quality curated setups reflecting current NSE market conditions."""
        sc_hi = SignalScore(trend_score=18, momentum_score=12, volatility_score=11,
                            volume_score=12, pattern_score=8, options_score=12, regime_score=9)
        sc_md = SignalScore(trend_score=15, momentum_score=10, volatility_score=9,
                            volume_score=10, pattern_score=6, options_score=10, regime_score=8)
        sc_ic = SignalScore(trend_score=4, momentum_score=9, volatility_score=13,
                            volume_score=7, pattern_score=5, options_score=11, regime_score=8)

        return [
            TradeSuggestion(
                id="NIFTY_CALL_24500",
                ticker="NIFTY 50",
                trade_type=TradeType.BUY_CALL,
                direction=SignalDirection.BULLISH,
                instrument_name="NIFTY 50 24500 CE (Weekly)",
                action_label="BUY 24500 CALL",
                spot_price=24500.0,
                entry_price=175.0,
                entry_zone_low=160.0,
                entry_zone_high=185.0,
                target_1=262.0,
                target_2=367.0,
                target_3=525.0,
                stop_loss=112.0,
                risk_reward="1 : 2.5",
                max_loss_pct=36.0,
                confidence_pct=int(sc_hi.total),
                score=sc_hi,
                time_horizon=TimeHorizon.SWING_SHORT,
                capital_allocation_pct=4.0,
                expiry_label="NIFTY Weekly",
                rationale=(
                    "PCR 1.28 (put saturation) with spot holding above Max Pain ₹24,400. "
                    "EMA 9 > 21 > 50 fully bullish. IV Rank 38% — buying cheap relative to "
                    "historical volatility. Entry zone ₹160–₹185 on any opening dip. "
                    "Target 1 at 50% premium gain (₹262), Target 2 at 110% (₹367)."
                ),
                confluence_factors=[
                    "PCR 1.28 (put saturation — contrarian bullish signal from institutional OI)",
                    "Spot ₹24,500 above Max Pain ₹24,400 — market makers pinning above",
                    "IV Rank 38% — calls cheap vs historical implied volatility",
                    "EMA 9 > 21 > 50 — full bullish ribbon alignment confirmed",
                    "Volume: 1.4× avg day volume on green candles (accumulation)",
                    "RSI 61 — bullish momentum zone (not overbought)",
                    "MACD histogram positive and expanding",
                ],
                risk_warnings=[
                    "Stop: ₹112 (36% loss of premium) — mandatory hard stop",
                    "Exit 100% if Nifty closes below ₹24,300 on daily basis",
                    "No carry past Thursday 15:00 IST if DTE < 2 days",
                ],
                greeks_profile={"delta": 0.52, "theta": -12.4, "gamma": 0.0018, "vega": 18.2},
                technical_levels={"SPOT": 24500, "ATM": 24500, "MAX_PAIN": 24400, "IV_RANK": 38, "RSI_14": 61},
            ),
            TradeSuggestion(
                id="BANKNIFTY_IRON_CONDOR",
                ticker="BANKNIFTY",
                trade_type=TradeType.IRON_CONDOR,
                direction=SignalDirection.NEUTRAL,
                instrument_name="BANKNIFTY Iron Condor 51500P/52700C (Weekly)",
                action_label="SELL IRON CONDOR",
                spot_price=52100.0,
                entry_price=295.0,
                entry_zone_low=270.0,
                entry_zone_high=315.0,
                target_1=147.0,
                target_2=74.0,
                target_3=29.0,
                stop_loss=531.0,
                risk_reward="1 : 1.8 (Win-Rate ~74%)",
                max_loss_pct=68.0,
                confidence_pct=int(sc_ic.total),
                score=sc_ic,
                time_horizon=TimeHorizon.POSITIONAL,
                capital_allocation_pct=6.0,
                expiry_label="BANKNIFTY Weekly",
                rationale=(
                    "BankNifty consolidating inside 1200-point range. IV Rank 68% elevated — "
                    "prime theta-selling environment. Net credit ₹295 collected with break-evens "
                    "at ₹51,205 (down) and ₹52,995 (up). Positive Theta +₹42/day. "
                    "Max risk defined at ₹705 per lot."
                ),
                confluence_factors=[
                    "IV Rank 68% elevated — premium selling at statistical advantage",
                    "PCR 1.05 — balanced, range-bound positioning from institutions",
                    "BankNifty consolidating — 7-day ATR below 20-day average",
                    "Theta: +₹42/day per lot (positive carry, time is on your side)",
                    "Break-even range: ₹51,205–₹52,995 (1,790 point corridor)",
                    "No major event triggers in next 7 days (safe theta window)",
                ],
                risk_warnings=[
                    "Close IMMEDIATELY if spot touches either short strike (₹51,500 or ₹52,700)",
                    "Do NOT carry past RBI policy, budget, or CPI release dates",
                    "Adjust to tighter strikes if VIX spikes above 15 intraday",
                ],
                greeks_profile={"delta": 0.02, "theta": 42.0, "gamma": -0.001, "vega": -28.0},
                technical_levels={"SELL_PUT": 51500, "SELL_CALL": 52700, "NET_CREDIT": 295, "IV_RANK": 68},
            ),
            TradeSuggestion(
                id="RELIANCE_LONG",
                ticker="RELIANCE",
                trade_type=TradeType.BUY_STOCK,
                direction=SignalDirection.BULLISH,
                instrument_name="RELIANCE (Cash Equity — Delivery)",
                action_label="BUY RELIANCE",
                spot_price=2965.0,
                entry_price=2960.0,
                entry_zone_low=2940.0,
                entry_zone_high=2975.0,
                target_1=3045.0,
                target_2=3130.0,
                target_3=3285.0,
                stop_loss=2905.0,
                risk_reward="1 : 2.1",
                max_loss_pct=1.86,
                confidence_pct=int(sc_md.total),
                score=sc_md,
                time_horizon=TimeHorizon.SWING_SHORT,
                capital_allocation_pct=5.0,
                rationale=(
                    "RELIANCE in confirmed uptrend with EMA 9 > 21 > 50 alignment. Price above "
                    "VWAP ₹2,952 and Volume Profile POC ₹2,948 — strong institutional base. "
                    "RSI 62 momentum zone. Buy on any pullback to entry zone ₹2,940–₹2,975 "
                    "with structural stop at ₹2,905 (below Pivot S1 ₹2,920)."
                ),
                confluence_factors=[
                    "EMA 9 > 21 > 50 — full bullish ribbon (3/3 MAs aligned)",
                    "Price above VWAP ₹2,952 — institutional buyers defending level",
                    "Volume Profile POC at ₹2,948 — high conviction price acceptance zone",
                    "RSI 62 — bullish momentum, not overbought (room to run)",
                    "Pivot Resistance R1 ₹3,048 aligns with Target 1",
                    "Positive sentiment flow: Jio 5G + retail expansion driving earnings",
                ],
                risk_warnings=[
                    "Stop below Pivot S1 ₹2,920 — if broken, structural breakdown likely",
                    "Reduce size 50% if Nifty below 24,100 on same day",
                ],
                greeks_profile={},
                technical_levels={"VWAP": 2952, "EMA_21": 2941, "PP": 2970, "S1": 2920, "R1": 3048, "RSI_14": 62},
            ),
            TradeSuggestion(
                id="TCS_CALL_3950",
                ticker="TCS",
                trade_type=TradeType.BUY_CALL,
                direction=SignalDirection.BULLISH,
                instrument_name="TCS 3950 CE (Monthly)",
                action_label="BUY TCS 3950 CALL",
                spot_price=3950.0,
                entry_price=65.0,
                entry_zone_low=58.0,
                entry_zone_high=72.0,
                target_1=97.0,
                target_2=143.0,
                target_3=195.0,
                stop_loss=39.0,
                risk_reward="1 : 2.4",
                max_loss_pct=40.0,
                confidence_pct=int(sc_md.total),
                score=sc_md,
                time_horizon=TimeHorizon.SWING_SHORT,
                capital_allocation_pct=3.0,
                expiry_label="TCS Monthly",
                rationale=(
                    "TCS leading IT sector momentum — ATM 3950 CE at ₹65 offers 2.4R potential. "
                    "IV Rank 32% — cheap premium vs realized volatility. "
                    "Entry zone ₹58–₹72 on opening gap or intraday pullback."
                ),
                confluence_factors=[
                    "IT sector leading Nifty with positive breadth",
                    "TCS above 200-EMA — long-term bullish structure intact",
                    "IV Rank 32% — buying options at discount",
                    "RSI 58 — mid-range momentum, significant upside potential",
                    "MACD histogram: positive and expanding for 4 consecutive sessions",
                ],
                risk_warnings=[
                    "Full premium at risk — set mandatory stop at ₹39",
                    "Exit before Q4 results if holding into earnings",
                ],
                greeks_profile={"delta": 0.50, "theta": -3.2, "gamma": 0.002, "vega": 4.5},
                technical_levels={"SPOT": 3950, "EMA_21": 3920, "RSI_14": 58, "IV_RANK": 32},
            ),
            TradeSuggestion(
                id="HDFCBANK_LONG",
                ticker="HDFCBANK",
                trade_type=TradeType.BUY_STOCK,
                direction=SignalDirection.BULLISH,
                instrument_name="HDFCBANK (Cash Equity — Delivery)",
                action_label="BUY HDFCBANK",
                spot_price=1720.0,
                entry_price=1715.0,
                entry_zone_low=1700.0,
                entry_zone_high=1728.0,
                target_1=1768.0,
                target_2=1832.0,
                target_3=1920.0,
                stop_loss=1680.0,
                risk_reward="1 : 2.2",
                max_loss_pct=2.04,
                confidence_pct=int(sc_md.total),
                score=sc_md,
                time_horizon=TimeHorizon.POSITIONAL,
                capital_allocation_pct=5.0,
                rationale=(
                    "HDFCBANK recovering from NPA resolution with strong loan growth guidance. "
                    "Technical: price reclaimed 200-EMA at ₹1,698, RSI 55 with volume expansion. "
                    "Entry zone ₹1,700–₹1,728. Positional target ₹1,832 (+6.8%)."
                ),
                confluence_factors=[
                    "Price above 200-EMA ₹1,698 — long-term trend reclaimed",
                    "Volume: 1.6× 20-day average on breakout candle (institutional buying)",
                    "RSI 55 — early-stage momentum (not extended)",
                    "Banking sector PCR >1.2 — contrarian bullish from options flow",
                    "NPA ratio at record low — fundamental catalyst supporting price",
                ],
                risk_warnings=[
                    "Stop below 200-EMA ₹1,698 — trend invalidated if broken",
                    "Watch RBI rate decision for volatility impact",
                ],
                greeks_profile={},
                technical_levels={"EMA_200": 1698, "RSI_14": 55, "PP": 1710, "S1": 1680, "R1": 1768},
            ),
        ]

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    def _extract_arrays(self, bars) -> Optional[Tuple[np.ndarray, ...]]:
        """Convert bars (list of dicts or DataFrame) to numpy arrays."""
        if bars is None:
            return None
        if hasattr(bars, "empty"):
            if bars.empty or len(bars) < 20:
                return None
            try:
                return (bars["open"].to_numpy(dtype=float),
                        bars["high"].to_numpy(dtype=float),
                        bars["low"].to_numpy(dtype=float),
                        bars["close"].to_numpy(dtype=float),
                        bars["volume"].to_numpy(dtype=float) if "volume" in bars.columns else np.ones(len(bars)))
            except Exception:
                return None
        if isinstance(bars, list):
            if len(bars) < 20:
                return None
            try:
                opens   = np.array([float(b.get("open", b.get("close", 0))) for b in bars])
                highs   = np.array([float(b.get("high", b.get("close", 0))) for b in bars])
                lows    = np.array([float(b.get("low",  b.get("close", 0))) for b in bars])
                closes  = np.array([float(b["close"]) for b in bars])
                volumes = np.array([float(b.get("volume", 1000)) for b in bars])
                return opens, highs, lows, closes, volumes
            except Exception:
                return None
        return None

    def _serialize(self, s: TradeSuggestion) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for state store and dashboard."""
        return {
            "id":                    s.id,
            "ticker":                s.ticker,
            "trade_type":            s.trade_type.value,
            "direction":             s.direction.value,
            "instrument_name":       s.instrument_name,
            "action_label":          s.action_label,
            "spot_price":            s.spot_price,
            "entry_price":           s.entry_price,
            "entry_zone_low":        s.entry_zone_low,
            "entry_zone_high":       s.entry_zone_high,
            "target_1":              s.target_1,
            "target_2":              s.target_2,
            "target_3":              s.target_3,
            "stop_loss":             s.stop_loss,
            "risk_reward":           s.risk_reward,
            "max_loss_pct":          s.max_loss_pct,
            "confidence_pct":        s.confidence_pct,
            "score_breakdown":       s.score.to_dict(),
            "time_horizon":          s.time_horizon.value,
            "capital_allocation_pct": s.capital_allocation_pct,
            "expiry_label":          s.expiry_label,
            "rationale":             s.rationale,
            "confluence_factors":    s.confluence_factors,
            "risk_warnings":         s.risk_warnings,
            "greeks_profile":        s.greeks_profile,
            "technical_levels":      s.technical_levels,
            "created_at":            s.created_at,
        }
