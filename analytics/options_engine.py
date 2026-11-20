"""
Institutional Options Pricing, Greeks & Derivatives Engine.
Engineered for Indian Indices (NIFTY, BANKNIFTY, FINNIFTY) & Equities.

Models:
    - Black-Scholes-Merton (1973) with Continuous Dividend Yield (Merton 1973)
    - Analytical First & Second Order Greeks: Delta, Gamma, Theta, Vega, Rho, Vanna, Volga
    - High-Precision Implied Volatility Solver (Newton-Raphson with Bisection fallback)
    - Volatility Smile & Skew Modeling
    - Put-Call Ratio (PCR) - Volume & Open Interest
    - Max Pain & Strike Pinning Analysis
    - Multi-Leg Strategy & Payoff Architecture (Iron Condor, Straddles, Butterfly, Spreads)
    - Portfolio Greeks Aggregation & Scenario Stress Testing (What-If Matrix)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import norm


class OptionType(str, Enum):
    CALL = "CE"
    PUT = "PE"


class OptionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OptionGreeks:
    price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0       # Rupee decay per calendar day
    vega: float = 0.0        # Rupee change per 1.0% change in IV
    rho: float = 0.0         # Rupee change per 1.0% change in interest rate
    vanna: float = 0.0       # Sensitivity of delta to volatility
    volga: float = 0.0       # Sensitivity of vega to volatility (vomma)
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0


@dataclass
class OptionStrikeData:
    strike: float
    # Call data
    call_ltp: float = 0.0
    call_iv: float = 0.0
    call_delta: float = 0.0
    call_gamma: float = 0.0
    call_theta: float = 0.0
    call_vega: float = 0.0
    call_oi: int = 0
    call_oi_change: int = 0
    call_volume: int = 0
    # Put data
    put_ltp: float = 0.0
    put_iv: float = 0.0
    put_delta: float = 0.0
    put_gamma: float = 0.0
    put_theta: float = 0.0
    put_vega: float = 0.0
    put_oi: int = 0
    put_oi_change: int = 0
    put_volume: int = 0


@dataclass
class OptionChainSnapshot:
    underlying: str
    spot_price: float
    atm_strike: float
    expiry_date: str
    dte_days: float
    risk_free_rate: float
    strikes: List[OptionStrikeData] = field(default_factory=list)
    pcr_oi: float = 0.0
    pcr_volume: float = 0.0
    max_pain_strike: float = 0.0
    atm_iv: float = 0.0
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class OptionLeg:
    strike: float
    option_type: OptionType
    action: OptionAction
    quantity: int = 1         # Lots / contracts
    premium: float = 0.0      # Entry price per unit
    iv: float = 0.15          # Implied vol for T+0 pricing
    lot_size: int = 50        # e.g., 50 for NIFTY, 15 for BANKNIFTY


@dataclass
class StrategyPayoffResult:
    name: str
    underlying: str
    spot_price: float
    net_premium: float        # Positive = Credit received, Negative = Debit paid
    max_profit: float
    max_loss: float
    risk_reward_ratio: str
    breakevens: List[float]
    probability_of_profit: float  # POP based on lognormal delta distribution
    margin_required: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    spot_range: List[float]
    payoff_expiry: List[float]
    payoff_t0: List[float]


class OptionsEngine:
    """
    Core mathematical derivatives and options desk pricing engine.
    """

    def __init__(self, risk_free_rate: float = 0.065, dividend_yield: float = 0.012) -> None:
        """
        :param risk_free_rate: Annualized risk-free rate (India RBI Repo/MIBOR approx 6.5%)
        :param dividend_yield: Annualized index/stock dividend yield (approx 1.2% for NIFTY)
        """
        self.r = risk_free_rate
        self.q = dividend_yield

    # ───────────────────────────────────────────────────────────────────
    # 1. Black-Scholes-Merton Analytical Model & Greeks
    # ───────────────────────────────────────────────────────────────────

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        dte_days: float,
        iv: float,
        option_type: OptionType,
    ) -> OptionGreeks:
        """
        Computes analytical European BSM Greeks for an option contract.
        """
        if spot <= 0 or strike <= 0 or iv <= 0:
            return OptionGreeks()

        # Handle expiration / 0-DTE boundary gracefully (minimum 15 minutes = ~0.00017 years)
        T = max(dte_days / 365.0, 1e-5)
        sigma = max(iv, 1e-4)

        sqrt_T = math.sqrt(T)
        d1 = (math.log(spot / strike) + (self.r - self.q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Cumulative & PDF distributions
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        N_prime_d1 = norm.pdf(d1)
        discount_r = math.exp(-self.r * T)
        discount_q = math.exp(-self.q * T)

        is_call = (option_type == OptionType.CALL)

        if is_call:
            # Call price & Greeks
            price = spot * discount_q * Nd1 - strike * discount_r * Nd2
            delta = discount_q * Nd1
            theta = (
                -(spot * discount_q * N_prime_d1 * sigma) / (2 * sqrt_T)
                - self.r * strike * discount_r * Nd2
                + self.q * spot * discount_q * Nd1
            ) / 365.0
            rho = (strike * T * discount_r * Nd2) / 100.0
            intrinsic = max(0.0, spot - strike)
        else:
            # Put price & Greeks
            N_neg_d1 = norm.cdf(-d1)
            N_neg_d2 = norm.cdf(-d2)
            price = strike * discount_r * N_neg_d2 - spot * discount_q * N_neg_d1
            delta = -discount_q * N_neg_d1
            theta = (
                -(spot * discount_q * N_prime_d1 * sigma) / (2 * sqrt_T)
                + self.r * strike * discount_r * N_neg_d2
                - self.q * spot * discount_q * N_neg_d1
            ) / 365.0
            rho = (-strike * T * discount_r * N_neg_d2) / 100.0
            intrinsic = max(0.0, strike - spot)

        # Gamma and Vega are identical for Calls and Puts
        gamma = (discount_q * N_prime_d1) / (spot * sigma * sqrt_T)
        vega = (spot * discount_q * N_prime_d1 * sqrt_T) / 100.0  # Per 1% change in vol

        # Second-order Greeks
        vanna = -(discount_q * N_prime_d1 * d2) / sigma
        volga = (vega * 100.0 * d1 * d2) / sigma

        price = max(price, intrinsic)
        extrinsic = max(0.0, price - intrinsic)

        return OptionGreeks(
            price=round(price, 2),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 2),
            vega=round(vega, 2),
            rho=round(rho, 4),
            vanna=round(vanna, 4),
            volga=round(volga, 4),
            intrinsic_value=round(intrinsic, 2),
            extrinsic_value=round(extrinsic, 2),
        )

    # ───────────────────────────────────────────────────────────────────
    # 2. Implied Volatility Solver (Newton-Raphson + Bisection Fallback)
    # ───────────────────────────────────────────────────────────────────

    def calculate_iv(
        self,
        market_price: float,
        spot: float,
        strike: float,
        dte_days: float,
        option_type: OptionType,
    ) -> float:
        """
        Extracts annualized Implied Volatility from market option price.
        """
        intrinsic = (
            max(0.0, spot - strike)
            if option_type == OptionType.CALL
            else max(0.0, strike - spot)
        )
        if market_price <= intrinsic or dte_days <= 0:
            return 0.15

        # Newton-Raphson iteration
        iv = 0.20  # initial guess 20%
        for _ in range(50):
            greeks = self.calculate_greeks(spot, strike, dte_days, iv, option_type)
            diff = greeks.price - market_price
            if abs(diff) < 0.01:
                return round(iv, 4)
            vega = greeks.vega * 100.0
            if vega < 1e-4:
                break
            iv_new = iv - diff / vega
            if iv_new <= 0.001 or iv_new > 4.0:
                break
            iv = iv_new

        # Bisection fallback
        low, high = 0.001, 3.5
        for _ in range(35):
            mid = (low + high) / 2.0
            greeks = self.calculate_greeks(spot, strike, dte_days, mid, option_type)
            if abs(greeks.price - market_price) < 0.02:
                return round(mid, 4)
            if greeks.price > market_price:
                high = mid
            else:
                low = mid

        return round((low + high) / 2.0, 4)

    # ───────────────────────────────────────────────────────────────────
    # 3. Volatility Skew & Realistic Option Chain Synthesis
    # ───────────────────────────────────────────────────────────────────

    def generate_option_chain(
        self,
        underlying: str,
        spot_price: float,
        atm_iv: float = 0.145,
        dte_days: float = 4.0,
        num_strikes: int = 15,
        strike_step: Optional[float] = None,
    ) -> OptionChainSnapshot:
        """
        Generates a comprehensive live-style Option Chain with strike ladder,
        calls, puts, Greeks, Open Interest (OI), Volume, and Volatility Skew.
        """
        # Determine appropriate strike step for Indian instruments
        if strike_step is None:
            if "BANKNIFTY" in underlying:
                strike_step = 100.0
            elif "NIFTY" in underlying:
                strike_step = 50.0
            elif spot_price > 5000:
                strike_step = 50.0
            elif spot_price > 1000:
                strike_step = 20.0
            else:
                strike_step = 10.0

        atm_strike = round(spot_price / strike_step) * strike_step
        min_strike = atm_strike - (num_strikes // 2) * strike_step
        strikes_list = [min_strike + i * strike_step for i in range(num_strikes)]

        chain_data: List[OptionStrikeData] = []
        total_call_oi = 0
        total_put_oi = 0
        total_call_vol = 0
        total_put_vol = 0

        # Base parameters for synthetic open interest distribution (lognormal around ATM)
        for k in strikes_list:
            moneyness = (k - spot_price) / spot_price
            
            # Volatility Skew: OTM Puts have higher IV (left skew / crash risk), OTM Calls slight smile
            # Formula: IV(K) = ATM_IV - skew_slope * moneyness + smile_curve * moneyness^2
            call_skew = atm_iv - 0.12 * moneyness + 0.35 * (moneyness**2)
            put_skew = atm_iv - 0.22 * moneyness + 0.45 * (moneyness**2)
            call_iv = max(0.08, min(call_skew, 0.95))
            put_iv = max(0.08, min(put_skew, 0.95))

            call_greeks = self.calculate_greeks(spot_price, k, dte_days, call_iv, OptionType.CALL)
            put_greeks = self.calculate_greeks(spot_price, k, dte_days, put_iv, OptionType.PUT)

            # Institutional Open Interest modeling (heavy clustering at round numbers / resistance)
            dist_factor = math.exp(-0.5 * ((k - atm_strike) / (2.5 * strike_step)) ** 2)
            base_oi = int(120_000 * dist_factor) + 5_000

            # Round strikes attract higher institutional writing
            round_bonus = 1.6 if (k % (strike_step * 5) == 0) else 1.0

            # Resistance strikes have high Call OI; Support strikes have high Put OI
            call_bias = 1.35 if k >= spot_price else 0.75
            put_bias = 1.40 if k <= spot_price else 0.70

            call_oi = int(base_oi * round_bonus * call_bias)
            put_oi = int(base_oi * round_bonus * put_bias)
            call_vol = int(call_oi * 0.42)
            put_vol = int(put_oi * 0.38)

            total_call_oi += call_oi
            total_put_oi += put_oi
            total_call_vol += call_vol
            total_put_vol += put_vol

            chain_data.append(
                OptionStrikeData(
                    strike=k,
                    call_ltp=call_greeks.price,
                    call_iv=round(call_iv * 100, 1),
                    call_delta=call_greeks.delta,
                    call_gamma=call_greeks.gamma,
                    call_theta=call_greeks.theta,
                    call_vega=call_greeks.vega,
                    call_oi=call_oi,
                    call_oi_change=int(call_oi * 0.08),
                    call_volume=call_vol,
                    put_ltp=put_greeks.price,
                    put_iv=round(put_iv * 100, 1),
                    put_delta=put_greeks.delta,
                    put_gamma=put_greeks.gamma,
                    put_theta=put_greeks.theta,
                    put_vega=put_greeks.vega,
                    put_oi=put_oi,
                    put_oi_change=int(put_oi * 0.09),
                    put_volume=put_vol,
                )
            )

        # Put-Call Ratio (PCR)
        pcr_oi = round(total_put_oi / max(total_call_oi, 1), 2)
        pcr_vol = round(total_put_vol / max(total_call_vol, 1), 2)

        # Max Pain Strike Calculation
        max_pain = self.calculate_max_pain(strikes_list, chain_data)

        # Expiry date estimation
        now = datetime.now()
        target_expiry = now.date()
        # Next Thursday for standard Indian weekly expiry
        days_ahead = (3 - now.weekday()) % 7
        if days_ahead == 0 and now.hour >= 15:
            days_ahead = 7
        expiry_str = (now.date()).strftime("%d-%b-%Y")

        return OptionChainSnapshot(
            underlying=underlying,
            spot_price=spot_price,
            atm_strike=atm_strike,
            expiry_date=expiry_str,
            dte_days=round(dte_days, 1),
            risk_free_rate=self.r,
            strikes=chain_data,
            pcr_oi=pcr_oi,
            pcr_volume=pcr_vol,
            max_pain_strike=max_pain,
            atm_iv=round(atm_iv * 100, 1),
            iv_rank=48.5,  # Mid-vol regime
            iv_percentile=52.0,
        )

    # ───────────────────────────────────────────────────────────────────
    # 4. Max Pain Strike & Pinning Calculation
    # ───────────────────────────────────────────────────────────────────

    def calculate_max_pain(
        self,
        strikes: List[float],
        chain_data: List[OptionStrikeData],
    ) -> float:
        """
        Computes the strike price at which option buyers lose the maximum amount of money
        (i.e., minimum payout from option writers at expiry).
        """
        if not strikes or not chain_data:
            return 0.0

        payout_by_strike: Dict[float, float] = {}

        for eval_strike in strikes:
            total_payout = 0.0
            for item in chain_data:
                # Call payout if spot ends at eval_strike
                call_loss = max(0.0, eval_strike - item.strike) * item.call_oi
                # Put payout if spot ends at eval_strike
                put_loss = max(0.0, item.strike - eval_strike) * item.put_oi
                total_payout += (call_loss + put_loss)
            payout_by_strike[eval_strike] = total_payout

        # Max pain is the strike with the minimum total payout
        min_strike = min(payout_by_strike.keys(), key=lambda k: payout_by_strike[k])
        return min_strike

    # ───────────────────────────────────────────────────────────────────
    # 5. Multi-Leg Strategy Payoff Visualizer & Metrics Engine
    # ───────────────────────────────────────────────────────────────────

    def evaluate_strategy_payoff(
        self,
        name: str,
        underlying: str,
        spot_price: float,
        dte_days: float,
        legs: List[OptionLeg],
    ) -> StrategyPayoffResult:
        """
        Evaluates a complex multi-leg options strategy.
        Computes:
            - Net Credit / Debit
            - Max Profit & Max Loss
            - Breakeven points
            - Exact Expiry Payoff line
            - Exact T+0 (Today's) Payoff curve using analytical BSM Greeks
            - Net Portfolio Greeks (Delta, Gamma, Theta, Vega)
            - Probability of Profit (POP)
        """
        if not legs:
            raise ValueError("At least one option leg is required")

        lot_size = legs[0].lot_size

        # Compute net premium:
        # BUY = -premium, SELL = +premium
        net_premium_per_unit = 0.0
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0

        for leg in legs:
            sign = 1.0 if leg.action == OptionAction.SELL else -1.0
            net_premium_per_unit += sign * leg.premium * leg.quantity

            # Current Greeks contribution
            g = self.calculate_greeks(spot_price, leg.strike, dte_days, leg.iv, leg.option_type)
            multiplier = sign * leg.quantity * lot_size
            net_delta += multiplier * g.delta
            net_gamma += multiplier * g.gamma
            net_theta += multiplier * g.theta
            net_vega += multiplier * g.vega

        net_premium_cash = net_premium_per_unit * lot_size

        # Spot range for payoff diagram (+/- 12% around spot)
        lower_bound = spot_price * 0.88
        upper_bound = spot_price * 1.12
        spot_points = np.linspace(lower_bound, upper_bound, 120).tolist()

        payoff_expiry: List[float] = []
        payoff_t0: List[float] = []

        for s in spot_points:
            # Payoff at Expiry
            p_exp = 0.0
            # Payoff at T+0 (Today)
            p_t0 = 0.0

            for leg in legs:
                sign = 1.0 if leg.action == OptionAction.SELL else -1.0
                qty = leg.quantity * lot_size

                # Intrinsic at expiry
                if leg.option_type == OptionType.CALL:
                    val_exp = max(0.0, s - leg.strike)
                else:
                    val_exp = max(0.0, leg.strike - s)

                # Net P&L for this leg at expiry
                leg_pnl_exp = sign * (val_exp - leg.premium) * qty
                p_exp += leg_pnl_exp

                # Value today (T+0) using BSM
                g_today = self.calculate_greeks(s, leg.strike, dte_days, leg.iv, leg.option_type)
                leg_pnl_t0 = sign * (g_today.price - leg.premium) * qty
                p_t0 += leg_pnl_t0

            payoff_expiry.append(round(p_exp, 2))
            payoff_t0.append(round(p_t0, 2))

        # Max Profit & Max Loss
        max_profit = max(payoff_expiry)
        max_loss = min(payoff_expiry)

        # Find Breakeven points where payoff cross 0
        breakevens: List[float] = []
        for i in range(len(spot_points) - 1):
            if (payoff_expiry[i] <= 0 and payoff_expiry[i + 1] >= 0) or (
                payoff_expiry[i] >= 0 and payoff_expiry[i + 1] <= 0
            ):
                # Linear interpolation
                s1, s2 = spot_points[i], spot_points[i + 1]
                p1, p2 = payoff_expiry[i], payoff_expiry[i + 1]
                if p2 != p1:
                    be = s1 - p1 * (s2 - s1) / (p2 - p1)
                    breakevens.append(round(be, 2))

        # Risk-to-Reward Ratio
        if abs(max_loss) > 1e-4:
            rr_val = abs(max_profit / max_loss)
            risk_reward = f"1 : {rr_val:.2f}" if rr_val >= 1 else f"{1/rr_val:.2f} : 1"
        else:
            risk_reward = "Unlimited"

        # Probability of Profit (POP) approximation using lognormal distribution
        # Based on fraction of spot range where payoff > 0 weighted by normal PDF
        sigma_dte = max(0.15 * math.sqrt(dte_days / 365.0), 0.01)
        pop_samples = 0
        profitable_samples = 0
        for s, p in zip(spot_points, payoff_expiry):
            weight = norm.pdf(math.log(s / spot_price) / sigma_dte)
            pop_samples += weight
            if p > 0:
                profitable_samples += weight

        pop = round((profitable_samples / max(pop_samples, 1e-5)) * 100, 1)

        # Margin Requirement with Hedging Benefit (Span + Exposure)
        # Defined risk spreads take the spread width; naked legs take 12% spot
        margin_required = self._calculate_margin(legs, spot_price, lot_size)

        return StrategyPayoffResult(
            name=name,
            underlying=underlying,
            spot_price=spot_price,
            net_premium=round(net_premium_cash, 2),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            risk_reward_ratio=risk_reward,
            breakevens=breakevens,
            probability_of_profit=pop,
            margin_required=round(margin_required, 2),
            net_delta=round(net_delta, 2),
            net_gamma=round(net_gamma, 4),
            net_theta=round(net_theta, 2),
            net_vega=round(net_vega, 2),
            spot_range=spot_points,
            payoff_expiry=payoff_expiry,
            payoff_t0=payoff_t0,
        )

    def _calculate_margin(self, legs: List[OptionLeg], spot: float, lot_size: int) -> float:
        """
        Calculates institutional margin with exchange-grade multi-leg spread benefits.
        """
        has_short = any(leg.action == OptionAction.SELL for leg in legs)
        if not has_short:
            # Net debit paid only
            debit = sum(leg.premium * leg.quantity * lot_size for leg in legs if leg.action == OptionAction.BUY)
            return debit

        # Check for defined risk spreads (e.g. Iron Condor / Vertical Spread)
        short_strikes = [leg.strike for leg in legs if leg.action == OptionAction.SELL]
        long_strikes = [leg.strike for leg in legs if leg.action == OptionAction.BUY]

        if long_strikes and short_strikes:
            # Defined-risk spread width
            max_spread = max(abs(s - l) for s in short_strikes for l in long_strikes)
            return max_spread * lot_size * 1.15  # Buffer margin

        # Naked option: approx 15% of contract value (Standard NSE SPAN + Exposure)
        return spot * lot_size * 0.15

    # ───────────────────────────────────────────────────────────────────
    # 6. Pre-Built Institutional Playbooks
    # ───────────────────────────────────────────────────────────────────

    def build_preset_strategy(
        self,
        strategy_preset: str,
        underlying: str,
        spot_price: float,
        dte_days: float = 4.0,
        atm_iv: float = 0.145,
    ) -> StrategyPayoffResult:
        """
        Constructs standard battle-tested institutional strategies.
        """
        step = 50.0 if "NIFTY" in underlying else (100.0 if "BANKNIFTY" in underlying else 20.0)
        lot = 50 if "NIFTY" in underlying else (15 if "BANKNIFTY" in underlying else 250)
        atm = round(spot_price / step) * step

        strat = strategy_preset.upper()

        if "IRON CONDOR" in strat:
            # 4-Leg: Sell OTM Put, Buy further OTM Put, Sell OTM Call, Buy further OTM Call
            sp = atm - 2 * step
            bp = atm - 4 * step
            sc = atm + 2 * step
            bc = atm + 4 * step

            p_sp = self.calculate_greeks(spot_price, sp, dte_days, atm_iv + 0.02, OptionType.PUT).price
            p_bp = self.calculate_greeks(spot_price, bp, dte_days, atm_iv + 0.03, OptionType.PUT).price
            p_sc = self.calculate_greeks(spot_price, sc, dte_days, atm_iv, OptionType.CALL).price
            p_bc = self.calculate_greeks(spot_price, bc, dte_days, atm_iv, OptionType.CALL).price

            legs = [
                OptionLeg(bp, OptionType.PUT, OptionAction.BUY, 1, p_bp, atm_iv + 0.03, lot),
                OptionLeg(sp, OptionType.PUT, OptionAction.SELL, 1, p_sp, atm_iv + 0.02, lot),
                OptionLeg(sc, OptionType.CALL, OptionAction.SELL, 1, p_sc, atm_iv, lot),
                OptionLeg(bc, OptionType.CALL, OptionAction.BUY, 1, p_bc, atm_iv, lot),
            ]
            return self.evaluate_strategy_payoff("Iron Condor (Delta-Neutral Theta Harvest)", underlying, spot_price, dte_days, legs)

        elif "IRON BUTTERFLY" in strat:
            # 4-Leg: Sell ATM Straddle, Buy wings
            bp = atm - 3 * step
            bc = atm + 3 * step

            p_atm_p = self.calculate_greeks(spot_price, atm, dte_days, atm_iv + 0.01, OptionType.PUT).price
            p_atm_c = self.calculate_greeks(spot_price, atm, dte_days, atm_iv, OptionType.CALL).price
            p_bp = self.calculate_greeks(spot_price, bp, dte_days, atm_iv + 0.03, OptionType.PUT).price
            p_bc = self.calculate_greeks(spot_price, bc, dte_days, atm_iv, OptionType.CALL).price

            legs = [
                OptionLeg(bp, OptionType.PUT, OptionAction.BUY, 1, p_bp, atm_iv + 0.03, lot),
                OptionLeg(atm, OptionType.PUT, OptionAction.SELL, 1, p_atm_p, atm_iv + 0.01, lot),
                OptionLeg(atm, OptionType.CALL, OptionAction.SELL, 1, p_atm_c, atm_iv, lot),
                OptionLeg(bc, OptionType.CALL, OptionAction.BUY, 1, p_bc, atm_iv, lot),
            ]
            return self.evaluate_strategy_payoff("Iron Butterfly (Pin Strike Harvest)", underlying, spot_price, dte_days, legs)

        elif "SHORT STRADDLE" in strat:
            # 2-Leg: Sell ATM Call + Sell ATM Put
            p_c = self.calculate_greeks(spot_price, atm, dte_days, atm_iv, OptionType.CALL).price
            p_p = self.calculate_greeks(spot_price, atm, dte_days, atm_iv + 0.01, OptionType.PUT).price
            legs = [
                OptionLeg(atm, OptionType.CALL, OptionAction.SELL, 1, p_c, atm_iv, lot),
                OptionLeg(atm, OptionType.PUT, OptionAction.SELL, 1, p_p, atm_iv + 0.01, lot),
            ]
            return self.evaluate_strategy_payoff("Short Straddle (Maximum Theta Decay)", underlying, spot_price, dte_days, legs)

        elif "LONG STRANGLE" in strat:
            # 2-Leg: Buy OTM Call + Buy OTM Put
            bp = atm - 2 * step
            bc = atm + 2 * step
            p_p = self.calculate_greeks(spot_price, bp, dte_days, atm_iv + 0.02, OptionType.PUT).price
            p_c = self.calculate_greeks(spot_price, bc, dte_days, atm_iv, OptionType.CALL).price
            legs = [
                OptionLeg(bp, OptionType.PUT, OptionAction.BUY, 1, p_p, atm_iv + 0.02, lot),
                OptionLeg(bc, OptionType.CALL, OptionAction.BUY, 1, p_c, atm_iv, lot),
            ]
            return self.evaluate_strategy_payoff("Long Strangle (Volatility Breakout / Event)", underlying, spot_price, dte_days, legs)

        elif "BEAR PUT SPREAD" in strat:
            # Buy ATM/OTM Put, Sell lower OTM Put
            bp = atm
            sp = atm - 2 * step
            p_bp = self.calculate_greeks(spot_price, bp, dte_days, atm_iv, OptionType.PUT).price
            p_sp = self.calculate_greeks(spot_price, sp, dte_days, atm_iv + 0.02, OptionType.PUT).price
            legs = [
                OptionLeg(bp, OptionType.PUT, OptionAction.BUY, 1, p_bp, atm_iv, lot),
                OptionLeg(sp, OptionType.PUT, OptionAction.SELL, 1, p_sp, atm_iv + 0.02, lot),
            ]
            return self.evaluate_strategy_payoff("Bear Put Spread (Defined-Risk Downside)", underlying, spot_price, dte_days, legs)

        else:  # Bull Call Spread default
            bc = atm
            sc = atm + 2 * step
            p_bc = self.calculate_greeks(spot_price, bc, dte_days, atm_iv, OptionType.CALL).price
            p_sc = self.calculate_greeks(spot_price, sc, dte_days, atm_iv, OptionType.CALL).price
            legs = [
                OptionLeg(bc, OptionType.CALL, OptionAction.BUY, 1, p_bc, atm_iv, lot),
                OptionLeg(sc, OptionType.CALL, OptionAction.SELL, 1, p_sc, atm_iv, lot),
            ]
            return self.evaluate_strategy_payoff("Bull Call Spread (Defined-Risk Upside)", underlying, spot_price, dte_days, legs)

    # ───────────────────────────────────────────────────────────────────
    # 7. Stress Testing & Scenario Risk Matrix
    # ───────────────────────────────────────────────────────────────────

    def scenario_stress_test(
        self,
        legs: List[OptionLeg],
        spot_price: float,
        dte_days: float,
        base_iv: float,
        spot_shocks_pct: Optional[List[float]] = None,
        iv_shocks_pct: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        30-Year Veteran Stress Test:
        Computes institutional P&L Heatmap across simultaneous Spot shocks and IV shocks.
        """
        if spot_shocks_pct is None:
            spot_shocks_pct = [-5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0]
        if iv_shocks_pct is None:
            iv_shocks_pct = [-25.0, -10.0, 0.0, 10.0, 25.0]

        lot_size = legs[0].lot_size if legs else 50
        matrix: List[List[float]] = []

        for iv_shock in iv_shocks_pct:
            row: List[float] = []
            test_iv = max(0.05, base_iv * (1.0 + iv_shock / 100.0))

            for spot_shock in spot_shocks_pct:
                test_spot = spot_price * (1.0 + spot_shock / 100.0)
                pnl = 0.0

                for leg in legs:
                    sign = 1.0 if leg.action == OptionAction.SELL else -1.0
                    g = self.calculate_greeks(test_spot, leg.strike, dte_days, test_iv, leg.option_type)
                    pnl += sign * (g.price - leg.premium) * leg.quantity * lot_size

                row.append(round(pnl, 2))
            matrix.append(row)

        return {
            "spot_shocks": [f"{s:+g}%" for s in spot_shocks_pct],
            "iv_shocks": [f"{v:+g}%" for v in iv_shocks_pct],
            "pnl_matrix": matrix,
        }
