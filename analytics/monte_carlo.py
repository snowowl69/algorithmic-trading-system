"""
Risk-of-Ruin Monte Carlo Simulator.

Runs many simulated equity paths using a given win-rate and
average win / average loss to estimate probability of ruin,
expected CAGR, and other risk metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def simulate_equity_paths(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    initial_equity: float,
    n_trades: int = 252,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> Dict[str, object]:
    """Run a Monte Carlo simulation of equity paths.

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade (0.0 - 1.0).
    avg_win : float
        Average percentage gain on a winning trade (e.g. 0.02 for 2 %).
    avg_loss : float
        Average percentage loss on a losing trade as a positive number
        (e.g. 0.01 for 1 %).  Internally applied as a negative return.
    initial_equity : float
        Starting equity for each simulation.
    n_trades : int
        Number of trades per simulation path (default 252).
    n_sims : int
        Number of independent simulations to run (default 10 000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``paths``              -- np.ndarray of shape (n_sims, n_trades) with
                                  equity after each trade.
        ``terminal_equities``  -- np.ndarray of final equities.
        ``prob_of_ruin_10pct`` -- P(max drawdown >= 10 %).
        ``prob_of_ruin_25pct`` -- P(max drawdown >= 25 %).
        ``prob_of_ruin_50pct`` -- P(max drawdown >= 50 %).
        ``median_terminal``    -- median terminal equity.
        ``percentile_5``       -- 5th percentile terminal equity.
        ``percentile_95``      -- 95th percentile terminal equity.
        ``expected_cagr``      -- mean annualised growth (assuming 252 trading
                                  days per year).
        ``max_drawdown_median``-- median of per-path maximum drawdowns.
    """
    rng = np.random.default_rng(seed)

    # Draw win/loss outcomes for every (sim, trade) pair
    outcomes = rng.random((n_sims, n_trades))
    returns = np.where(outcomes < win_rate, avg_win, -abs(avg_loss))

    # Compound equity trade by trade
    multipliers = 1.0 + returns  # shape (n_sims, n_trades)
    cum_mult = np.cumprod(multipliers, axis=1)
    paths = initial_equity * cum_mult

    terminal_equities = paths[:, -1]

    # -- running max and drawdown per path --
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / np.maximum(running_max, 1e-10)
    max_dd_per_path = drawdowns.max(axis=1)  # shape (n_sims,)

    prob_of_ruin_10 = float(np.mean(max_dd_per_path >= 0.10))
    prob_of_ruin_25 = float(np.mean(max_dd_per_path >= 0.25))
    prob_of_ruin_50 = float(np.mean(max_dd_per_path >= 0.50))

    median_terminal = float(np.median(terminal_equities))
    pct_5 = float(np.percentile(terminal_equities, 5))
    pct_95 = float(np.percentile(terminal_equities, 95))

    # CAGR: treat n_trades as 252 trading days = 1 year
    years = n_trades / 252.0
    total_returns = terminal_equities / initial_equity
    # Avoid negative or zero returns in power
    safe_returns = np.maximum(total_returns, 1e-10)
    cagrs = safe_returns ** (1.0 / max(years, 1e-10)) - 1.0
    expected_cagr = float(np.mean(cagrs))

    max_dd_median = float(np.median(max_dd_per_path))

    return {
        "paths": paths,
        "terminal_equities": terminal_equities,
        "prob_of_ruin_10pct": round(prob_of_ruin_10, 4),
        "prob_of_ruin_25pct": round(prob_of_ruin_25, 4),
        "prob_of_ruin_50pct": round(prob_of_ruin_50, 4),
        "median_terminal": round(median_terminal, 2),
        "percentile_5": round(pct_5, 2),
        "percentile_95": round(pct_95, 2),
        "expected_cagr": round(expected_cagr, 6),
        "max_drawdown_median": round(max_dd_median, 4),
    }
