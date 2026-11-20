"""
Correlation and Portfolio Risk Analytics.

Provides correlation-matrix computation, concentration metrics,
sector exposure (Indian equity sectors), and portfolio beta.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Indian equity sector mapping
# ------------------------------------------------------------------

_SECTOR_MAP: Dict[str, str] = {
    "RELIANCE":    "Energy",
    "ADANIENT":    "Energy",
    "TCS":         "IT",
    "INFY":        "IT",
    "WIPRO":       "IT",
    "HDFCBANK":    "Banking",
    "ICICIBANK":   "Banking",
    "SBIN":        "Banking",
    "AXISBANK":    "Banking",
    "KOTAKBANK":   "Banking",
    "BAJFINANCE":  "Banking",
    "MARUTI":      "Auto",
    "TATAMOTORS":  "Auto",
    "ITC":         "FMCG",
    "HINDUNILVR":  "FMCG",
    "NESTLEIND":   "FMCG",
    "BHARTIARTL":  "Telecom",
    "LT":          "Infra",
    "SUNPHARMA":   "Pharma",
    "TITAN":       "Consumer",
}


def _extract_close_series(bars: List[Dict[str, Any]]) -> np.ndarray:
    """Extract an array of close prices from a list of bar dicts."""
    return np.array([float(b["close"]) for b in bars], dtype=float)


def compute_correlation_matrix(bars_dict: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Compute the pairwise correlation matrix of daily returns.

    Parameters
    ----------
    bars_dict : dict
        Mapping of ``{ticker: [bar_dict, ...]}`` where each bar dict has
        at least a ``"close"`` key.

    Returns
    -------
    pd.DataFrame
        Correlation matrix indexed and columned by ticker.
    """
    returns: Dict[str, np.ndarray] = {}
    for ticker, bars in bars_dict.items():
        closes = _extract_close_series(bars)
        if len(closes) < 2:
            continue
        pct = np.diff(closes) / np.maximum(np.abs(closes[:-1]), 1e-10)
        returns[ticker] = pct

    if not returns:
        return pd.DataFrame()

    # Align to shortest common length
    min_len = min(len(r) for r in returns.values())
    df = pd.DataFrame({t: r[-min_len:] for t, r in returns.items()})
    return df.corr()


def compute_portfolio_metrics(
    bars_dict: Dict[str, List[Dict[str, Any]]],
    positions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute portfolio-level risk metrics.

    Parameters
    ----------
    bars_dict : dict
        ``{ticker: [bar_dict, ...]}`` for universe of instruments.
    positions : list[dict]
        Each dict must contain at least ``Ticker`` (str), ``Current``
        (float, current price), and ``Qty`` (float, quantity).

    Returns
    -------
    dict
        ``correlation_matrix``    -- pd.DataFrame
        ``herfindahl_index``      -- float (0-1 scale; 1 = single stock)
        ``max_concentration_pct`` -- float (largest position weight, 0-100)
        ``sector_exposure``       -- dict mapping sector -> weight (0-100)
        ``portfolio_beta``        -- float (vs equal-weight benchmark)
    """
    corr = compute_correlation_matrix(bars_dict)

    # -- position weights --
    market_values: Dict[str, float] = {}
    for pos in positions:
        ticker = pos.get("Ticker", pos.get("ticker", ""))
        price = float(pos.get("Current", pos.get("current_price", 0.0)))
        qty = float(pos.get("Qty", pos.get("quantity", 0.0)))
        mv = abs(price * qty)
        market_values[ticker] = market_values.get(ticker, 0.0) + mv

    total_mv = sum(market_values.values()) or 1.0
    weights = {t: v / total_mv for t, v in market_values.items()}

    # Herfindahl index (sum of squared weights)
    hhi = sum(w * w for w in weights.values())

    # Max concentration
    max_conc = max(weights.values(), default=0.0) * 100.0

    # Sector exposure
    sector_weights: Dict[str, float] = {}
    for ticker, w in weights.items():
        sector = _SECTOR_MAP.get(ticker.upper(), "Other")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w * 100.0

    # Portfolio beta vs equal-weight benchmark of all tickers in bars_dict
    portfolio_beta = _compute_portfolio_beta(bars_dict, weights)

    return {
        "correlation_matrix": corr,
        "herfindahl_index": round(hhi, 6),
        "max_concentration_pct": round(max_conc, 2),
        "sector_exposure": {s: round(v, 2) for s, v in sorted(sector_weights.items())},
        "portfolio_beta": round(portfolio_beta, 4),
    }


def _compute_portfolio_beta(
    bars_dict: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float],
) -> float:
    """Estimate portfolio beta against an equal-weight benchmark of all tickers."""
    # Gather per-ticker return series
    returns_map: Dict[str, np.ndarray] = {}
    for ticker, bars in bars_dict.items():
        closes = _extract_close_series(bars)
        if len(closes) < 2:
            continue
        pct = np.diff(closes) / np.maximum(np.abs(closes[:-1]), 1e-10)
        returns_map[ticker] = pct

    if len(returns_map) < 2:
        return 1.0

    min_len = min(len(r) for r in returns_map.values())
    if min_len < 2:
        return 1.0

    # Build benchmark (equal-weight average return)
    all_returns = np.column_stack([r[-min_len:] for r in returns_map.values()])
    benchmark = all_returns.mean(axis=1)

    # Build weighted portfolio return
    tickers_ordered = list(returns_map.keys())
    port_ret = np.zeros(min_len)
    total_w = 0.0
    for idx, ticker in enumerate(tickers_ordered):
        w = weights.get(ticker, 0.0)
        port_ret += w * returns_map[ticker][-min_len:]
        total_w += w

    if total_w > 0.0:
        port_ret /= total_w

    var_bench = float(np.var(benchmark, ddof=1))
    if var_bench < 1e-14:
        return 1.0

    cov = float(np.cov(port_ret, benchmark, ddof=1)[0, 1])
    return cov / var_bench
