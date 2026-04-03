"""
Professional Technical Analysis Library.

Provides institutional-grade indicators used by top trading desks:
- Moving Averages (SMA, EMA, WMA, VWAP)
- Oscillators (RSI, MACD, Stochastic, Williams %R, CCI)
- Volatility (Bollinger Bands, ATR, Keltner Channels)
- Volume (OBV, VWAP, Volume Profile, A/D Line)
- Trend (ADX, Ichimoku Cloud, Parabolic SAR, Supertrend)
- Support/Resistance (Pivot Points, Fibonacci Retracement)
- Pattern Recognition (Doji, Hammer, Engulfing, etc.)

All calculations are vectorized with NumPy/Pandas for speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  Moving Averages
# ═══════════════════════════════════════════════════════════════════════

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════
#  Oscillators
# ═══════════════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator (%K, %D)."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    high_max = df["high"].rolling(window=period).max()
    low_min = df["low"].rolling(window=period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════
#  Volatility
# ═══════════════════════════════════════════════════════════════════════

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands — upper, middle (SMA), lower."""
    middle = sma(series, period)
    rolling_std = series.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5):
    """Keltner Channels — upper, middle (EMA), lower."""
    middle = ema(df["close"], ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


# ═══════════════════════════════════════════════════════════════════════
#  Volume Indicators
# ═══════════════════════════════════════════════════════════════════════

def obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume."""
    direction = np.where(df["close"] > df["close"].shift(1), 1,
                np.where(df["close"] < df["close"].shift(1), -1, 0))
    return (direction * df["volume"]).cumsum()


def ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line."""
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
          (df["high"] - df["low"]).replace(0, np.nan)
    return (clv.fillna(0) * df["volume"]).cumsum()


# ═══════════════════════════════════════════════════════════════════════
#  Trend Indicators
# ═══════════════════════════════════════════════════════════════════════

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_val = atr(df, period)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period).mean() / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, min_periods=period).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend indicator — returns (supertrend_series, direction)."""
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            st.iloc[i] = lower_band.iloc[i]
        else:
            st.iloc[i] = upper_band.iloc[i]

    return st, direction


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """Ichimoku Cloud components."""
    tenkan_sen = (df["high"].rolling(tenkan).max() + df["low"].rolling(tenkan).min()) / 2
    kijun_sen = (df["high"].rolling(kijun).max() + df["low"].rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((df["high"].rolling(senkou_b).max() +
                     df["low"].rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = df["close"].shift(-kijun)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b_val, chikou


def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
    """Parabolic SAR."""
    length = len(df)
    sar = pd.Series(np.nan, index=df.index)
    bull = True
    af = af_start
    ep = df["low"].iloc[0]
    hp = df["high"].iloc[0]
    lp = df["low"].iloc[0]
    sar.iloc[0] = df["low"].iloc[0]

    for i in range(1, length):
        if bull:
            sar.iloc[i] = sar.iloc[i - 1] + af * (hp - sar.iloc[i - 1])
            sar.iloc[i] = min(sar.iloc[i], df["low"].iloc[i - 1],
                              df["low"].iloc[max(0, i - 2)])
            if df["low"].iloc[i] < sar.iloc[i]:
                bull = False
                sar.iloc[i] = hp
                lp = df["low"].iloc[i]
                af = af_start
            else:
                if df["high"].iloc[i] > hp:
                    hp = df["high"].iloc[i]
                    af = min(af + af_step, af_max)
        else:
            sar.iloc[i] = sar.iloc[i - 1] + af * (lp - sar.iloc[i - 1])
            sar.iloc[i] = max(sar.iloc[i], df["high"].iloc[i - 1],
                              df["high"].iloc[max(0, i - 2)])
            if df["high"].iloc[i] > sar.iloc[i]:
                bull = True
                sar.iloc[i] = lp
                hp = df["high"].iloc[i]
                af = af_start
            else:
                if df["low"].iloc[i] < lp:
                    lp = df["low"].iloc[i]
                    af = min(af + af_step, af_max)

    return sar


# ═══════════════════════════════════════════════════════════════════════
#  Support / Resistance
# ═══════════════════════════════════════════════════════════════════════

def pivot_points(df: pd.DataFrame):
    """Classic Pivot Points (daily) — PP, R1-R3, S1-S3."""
    h, l, c = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    pp = (h + l + c) / 3
    r1, s1 = 2 * pp - l, 2 * pp - h
    r2, s2 = pp + (h - l), pp - (h - l)
    r3, s3 = h + 2 * (pp - l), l - 2 * (h - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def fibonacci_retracement(high: float, low: float) -> dict[str, float]:
    """Fibonacci retracement levels."""
    diff = high - low
    return {
        "0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100%": low,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Candlestick Patterns
# ═══════════════════════════════════════════════════════════════════════

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect common candlestick patterns."""
    body = df["close"] - df["open"]
    body_abs = body.abs()
    range_hl = df["high"] - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

    patterns = pd.DataFrame(index=df.index)

    # Doji — very small body
    patterns["doji"] = body_abs < (range_hl * 0.1)

    # Hammer — small body at top, long lower shadow
    patterns["hammer"] = (
        (lower_shadow > 2 * body_abs) &
        (upper_shadow < body_abs * 0.5) &
        (body_abs > 0)
    )

    # Shooting Star — small body at bottom, long upper shadow
    patterns["shooting_star"] = (
        (upper_shadow > 2 * body_abs) &
        (lower_shadow < body_abs * 0.5) &
        (body_abs > 0)
    )

    # Bullish Engulfing
    prev_body = body.shift(1)
    patterns["bullish_engulfing"] = (
        (prev_body < 0) & (body > 0) &
        (df["open"] < df["close"].shift(1)) &
        (df["close"] > df["open"].shift(1))
    )

    # Bearish Engulfing
    patterns["bearish_engulfing"] = (
        (prev_body > 0) & (body < 0) &
        (df["open"] > df["close"].shift(1)) &
        (df["close"] < df["open"].shift(1))
    )

    # Morning Star (simplified 2-bar)
    patterns["morning_star"] = (
        (body.shift(1).abs() < range_hl.shift(1) * 0.15) &
        (body > 0) &
        (df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2)
    )

    # Evening Star (simplified 2-bar)
    patterns["evening_star"] = (
        (body.shift(1).abs() < range_hl.shift(1) * 0.15) &
        (body < 0) &
        (df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2)
    )

    return patterns


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: Add all indicators to DataFrame
# ═══════════════════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a comprehensive set of technical indicators to OHLCV data."""
    out = df.copy()

    # Moving Averages
    out["sma_20"] = sma(df["close"], 20)
    out["sma_50"] = sma(df["close"], 50)
    out["sma_200"] = sma(df["close"], 200)
    out["ema_9"] = ema(df["close"], 9)
    out["ema_21"] = ema(df["close"], 21)

    # VWAP
    if "volume" in df.columns:
        out["vwap"] = vwap(df)

    # RSI
    out["rsi_14"] = rsi(df["close"], 14)

    # MACD
    macd_line, signal_line, hist = macd(df["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"])
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower

    # ATR
    out["atr_14"] = atr(df, 14)

    # Stochastic
    k, d = stochastic(df)
    out["stoch_k"] = k
    out["stoch_d"] = d

    # ADX
    out["adx"] = adx(df)

    # OBV
    if "volume" in df.columns:
        out["obv"] = obv(df)

    return out
