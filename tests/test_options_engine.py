"""
Unit Tests for Institutional Options Pricing, Greeks, Volatility, & Payoff Engine.
"""

import pytest
from analytics.options_engine import (
    OptionsEngine, OptionType, OptionAction, OptionLeg
)
from data_engine.options_feed import LiveOptionsFeed


class TestOptionsPricingAndGreeks:
    def setup_method(self):
        self.engine = OptionsEngine(risk_free_rate=0.065, dividend_yield=0.012)

    def test_call_put_parity(self):
        """Verify Black-Scholes Call-Put parity: C - P = S*e^(-qT) - K*e^(-rT)."""
        spot = 24500.0
        strike = 24500.0
        dte = 30.0
        iv = 0.15
        T = dte / 365.0
        import math

        call_g = self.engine.calculate_greeks(spot, strike, dte, iv, OptionType.CALL)
        put_g = self.engine.calculate_greeks(spot, strike, dte, iv, OptionType.PUT)

        lhs = call_g.price - put_g.price
        rhs = spot * math.exp(-self.engine.q * T) - strike * math.exp(-self.engine.r * T)
        assert abs(lhs - rhs) < 0.25

    def test_delta_ranges(self):
        """Calls must have Delta in (0, 1) and Puts in (-1, 0)."""
        spot = 24500.0
        call_atm = self.engine.calculate_greeks(spot, 24500.0, 10.0, 0.15, OptionType.CALL)
        put_atm = self.engine.calculate_greeks(spot, 24500.0, 10.0, 0.15, OptionType.PUT)

        assert 0.40 < call_atm.delta < 0.60
        assert -0.60 < put_atm.delta < -0.40
        assert call_atm.gamma > 0
        assert put_atm.gamma > 0
        assert call_atm.theta < 0  # Time decay is negative for long options
        assert call_atm.vega > 0   # Long options benefit from rising IV

    def test_iv_solver(self):
        """IV solver must accurately invert BSM prices."""
        spot = 24500.0
        strike = 24600.0
        true_iv = 0.18
        dte = 14.0

        greeks = self.engine.calculate_greeks(spot, strike, dte, true_iv, OptionType.CALL)
        solved_iv = self.engine.calculate_iv(greeks.price, spot, strike, dte, OptionType.CALL)
        assert abs(solved_iv - true_iv) < 0.01

    def test_option_chain_generation(self):
        """Option chain must compute valid PCR, strikes, and Max Pain."""
        chain = self.engine.generate_option_chain("NIFTY 50", 24500.0, atm_iv=0.14, dte_days=4.0, num_strikes=15)
        assert chain.underlying == "NIFTY 50"
        assert chain.atm_strike == 24500.0
        assert len(chain.strikes) == 15
        assert chain.pcr_oi > 0.1
        assert chain.max_pain_strike > 0.0

    def test_iron_condor_preset(self):
        """Iron Condor must yield defined risk (max_loss < infinity) and positive net credit."""
        res = self.engine.build_preset_strategy("IRON CONDOR", "NIFTY 50", 24500.0, dte_days=4.0)
        assert res.net_premium > 0  # Net credit received
        assert res.max_profit > 0
        assert res.max_loss < 0
        assert len(res.breakevens) == 2  # Lower and upper breakeven
        assert 40.0 < res.probability_of_profit < 90.0

    def test_scenario_stress_test(self):
        """Stress matrix must return 2D grid of P&L outcomes."""
        lot = 50
        legs = [
            OptionLeg(24500.0, OptionType.CALL, OptionAction.SELL, 1, 150.0, 0.15, lot),
            OptionLeg(24500.0, OptionType.PUT, OptionAction.SELL, 1, 140.0, 0.15, lot),
        ]
        stress = self.engine.scenario_stress_test(legs, 24500.0, 4.0, 0.15)
        assert len(stress["spot_shocks"]) == 7
        assert len(stress["iv_shocks"]) == 5
        assert len(stress["pnl_matrix"]) == 5
        assert len(stress["pnl_matrix"][0]) == 7


class TestLiveOptionsFeed:
    def test_feed_snapshot(self):
        feed = LiveOptionsFeed()
        snap = feed.get_snapshot()
        assert "NIFTY 50" in snap
        assert "BANKNIFTY" in snap
        assert snap["NIFTY 50"]["spot_price"] > 0
        assert len(snap["NIFTY 50"]["strikes"]) > 0
