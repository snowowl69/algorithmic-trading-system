"""
Trade Analytics Engine — computes profit/loss stats, win rates, per-trade
breakdowns, strategy comparisons, and everything needed for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeAnalytics:
    """Compute analytics from a list of closed-trade dicts."""

    trades: List[Dict[str, Any]] = field(default_factory=list)

    # ── Core stats ────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a full P&L summary dictionary."""
        if not self.trades:
            return self._empty_summary()

        pnls = [t.get("pnl", 0.0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        even = [p for p in pnls if p == 0]

        total_pnl = sum(pnls)
        gross_profit = sum(wins)
        gross_loss = sum(losses)

        win_count = len(wins)
        loss_count = len(losses)
        total = len(pnls)
        win_rate = win_count / total if total > 0 else 0.0
        loss_rate = loss_count / total if total > 0 else 0.0

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_trade = np.mean(pnls) if pnls else 0.0
        median_trade = float(np.median(pnls)) if pnls else 0.0

        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0

        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

        # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Consecutive wins / losses
        streak_win, streak_loss = self._streaks(pnls)

        return {
            "total_trades": total,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "even_trades": len(even),
            "win_rate_pct": round(win_rate * 100, 2),
            "loss_rate_pct": round(loss_rate * 100, 2),
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 4),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "avg_trade": round(float(avg_trade), 2),
            "median_trade": round(median_trade, 2),
            "max_win": round(max_win, 2),
            "max_loss": round(max_loss, 2),
            "expectancy": round(expectancy, 2),
            "max_consecutive_wins": streak_win,
            "max_consecutive_losses": streak_loss,
        }

    # ── Risk-adjusted metrics ─────────────────────────────────────────

    def risk_metrics(
        self,
        risk_free_rate: float = 0.06,
        equity_curve: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Compute institutional-grade risk-adjusted performance metrics."""
        pnls = np.array([t.get("pnl", 0.0) for t in self.trades])
        if len(pnls) < 2:
            return {
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
                "calmar_ratio": 0.0, "max_drawdown_pct": 0.0,
                "max_drawdown_duration": 0,
            }

        # Annualise assuming ~252 trading days
        mean_ret = float(np.mean(pnls))
        std_ret = float(np.std(pnls, ddof=1))
        daily_rf = risk_free_rate / 252

        # Sharpe Ratio
        sharpe = ((mean_ret - daily_rf) / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # Sortino Ratio — penalise only downside deviation
        downside = pnls[pnls < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1.0
        sortino = ((mean_ret - daily_rf) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        # Max drawdown from equity curve
        max_dd_pct = 0.0
        max_dd_duration = 0
        if equity_curve and len(equity_curve) > 1:
            eq = np.array(equity_curve, dtype=float)
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
            max_dd_pct = float(np.max(dd)) * 100

            # Duration: longest streak below peak
            below_peak = eq < peak
            dur = 0
            for b in below_peak:
                if b:
                    dur += 1
                    max_dd_duration = max(max_dd_duration, dur)
                else:
                    dur = 0

        # Calmar Ratio
        total_return_pct = float(np.sum(pnls))
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "max_drawdown_duration": max_dd_duration,
        }

    # ── Per-strategy breakdown ────────────────────────────────────────

    def by_strategy(self) -> Dict[str, Dict[str, Any]]:
        """Return summary stats grouped by strategy name."""
        groups: Dict[str, list] = {}
        for t in self.trades:
            strat = t.get("strategy", "unknown")
            groups.setdefault(strat, []).append(t)
        return {
            name: TradeAnalytics(trades=tlist).summary()
            for name, tlist in groups.items()
        }

    # ── Per-ticker breakdown ──────────────────────────────────────────

    def by_ticker(self) -> Dict[str, Dict[str, Any]]:
        """Return summary stats grouped by ticker."""
        groups: Dict[str, list] = {}
        for t in self.trades:
            tk = t.get("ticker", "???")
            groups.setdefault(tk, []).append(t)
        return {
            name: TradeAnalytics(trades=tlist).summary()
            for name, tlist in groups.items()
        }

    # ── DataFrames for charts ─────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to a DataFrame with computed columns."""
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        if "pnl" in df.columns:
            df["cumulative_pnl"] = df["pnl"].cumsum()
            df["trade_number"] = range(1, len(df) + 1)
            df["result"] = df["pnl"].apply(
                lambda x: "Win" if x > 0 else ("Loss" if x < 0 else "Even")
            )
            df["pnl_pct"] = df.apply(
                lambda r: round(r["pnl"] / (r["entry"] * r["qty"]) * 100, 2)
                if r.get("entry", 0) > 0 and r.get("qty", 0) > 0 else 0.0,
                axis=1,
            )
        return df

    def pnl_by_ticker_df(self) -> pd.DataFrame:
        """Aggregate PnL by ticker."""
        if not self.trades:
            return pd.DataFrame(columns=["ticker", "total_pnl", "trade_count", "win_rate_pct"])
        df = pd.DataFrame(self.trades)
        agg = df.groupby("ticker").agg(
            total_pnl=("pnl", "sum"),
            trade_count=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
            wins=("pnl", lambda x: (x > 0).sum()),
        ).reset_index()
        agg["win_rate_pct"] = round(agg["wins"] / agg["trade_count"] * 100, 1)
        return agg.drop(columns=["wins"])

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _streaks(pnls: List[float]) -> tuple[int, int]:
        max_w = max_l = cur_w = cur_l = 0
        for p in pnls:
            if p > 0:
                cur_w += 1
                cur_l = 0
            elif p < 0:
                cur_l += 1
                cur_w = 0
            else:
                cur_w = cur_l = 0
            max_w = max(max_w, cur_w)
            max_l = max(max_l, cur_l)
        return max_w, max_l

    @staticmethod
    def _empty_summary() -> Dict[str, Any]:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "even_trades": 0, "win_rate_pct": 0.0, "loss_rate_pct": 0.0,
            "total_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "avg_trade": 0.0, "median_trade": 0.0, "max_win": 0.0,
            "max_loss": 0.0, "expectancy": 0.0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        }
