"""
Unit tests for new institutional modules:
- Market Regime Detector
- Session Manager & ORB
- Monte Carlo Simulator
- Portfolio Risk & Correlation
- VWAP Strategy
- Options Signal Strategy
"""

from datetime import datetime, time, timedelta, timezone
import numpy as np
import pytest

from analytics.market_regime import MarketRegimeDetector, MarketRegime
from analytics.session_manager import (
    IndianSession, get_current_session, get_session_info, ORBTracker, _IST
)
from analytics.monte_carlo import simulate_equity_paths
from analytics.portfolio_risk import (
    compute_correlation_matrix, compute_portfolio_metrics
)
from core.models import Bar, SignalType
from strategies.vwap_strategy import VWAPStrategy
from strategies.options_signal_strategy import OptionsSignalStrategy


# ═══════════════════════════════════════════════════════════════════════
#  Market Regime Tests
# ═══════════════════════════════════════════════════════════════════════

def test_market_regime_insufficient_data():
    detector = MarketRegimeDetector()
    res = detector.detect(np.array([100.0, 101.0]), np.array([102.0, 102.0]), np.array([99.0, 100.0]))
    assert "regime" in res


def test_market_regime_classification():
    detector = MarketRegimeDetector()
    # Strongly trending up series
    closes = np.linspace(100, 150, 40)
    highs = closes + 1.0
    lows = closes - 1.0
    res = detector.detect(closes, highs, lows)
    assert res["regime"] in [MarketRegime.TRENDING_UP, MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY, MarketRegime.SQUEEZE]
    assert "adx_value" in res
    assert "bb_width" in res


# ═══════════════════════════════════════════════════════════════════════
#  Session Manager & ORB Tests
# ═══════════════════════════════════════════════════════════════════════

def test_indian_session_classification():
    # Test morning session at 10:30 IST
    t_morning = datetime(2026, 7, 3, 10, 30, tzinfo=_IST)
    assert get_current_session(t_morning) == IndianSession.MORNING

    # Test lunch at 12:30 IST
    t_lunch = datetime(2026, 7, 3, 12, 30, tzinfo=_IST)
    info_lunch = get_session_info(t_lunch)
    assert info_lunch["session"] == IndianSession.LUNCH
    assert info_lunch["should_avoid_entry"] is True

    # Test ORB window at 09:20 IST
    t_orb = datetime(2026, 7, 3, 9, 20, tzinfo=_IST)
    info_orb = get_session_info(t_orb)
    assert info_orb["is_orb_window"] is True


def test_orb_tracker():
    tracker = ORBTracker()
    day = datetime(2026, 7, 3, tzinfo=_IST)

    # 09:16 bar
    tracker.update(105.0, 95.0, day.replace(hour=9, minute=16))
    # 09:25 bar higher high
    tracker.update(110.0, 93.0, day.replace(hour=9, minute=25))
    assert not tracker.is_complete
    assert tracker.orb_high == 110.0
    assert tracker.orb_low == 93.0

    # 09:35 bar (past ORB window)
    tracker.update(108.0, 106.0, day.replace(hour=9, minute=35))
    assert tracker.is_complete

    # Breakout check
    assert tracker.check_breakout(112.0) == "breakout_above"
    assert tracker.check_breakout(90.0) == "breakout_below"
    assert tracker.check_breakout(100.0) is None


# ═══════════════════════════════════════════════════════════════════════
#  Monte Carlo Simulator Tests
# ═══════════════════════════════════════════════════════════════════════

def test_monte_carlo_simulation():
    res = simulate_equity_paths(
        win_rate=0.55,
        avg_win=0.03,
        avg_loss=0.015,
        initial_equity=1_000_000.0,
        n_trades=50,
        n_sims=500,
        seed=42,
    )
    assert res["paths"].shape == (500, 50)
    assert res["median_terminal"] > 0
    assert 0.0 <= res["prob_of_ruin_10pct"] <= 1.0
    assert res["percentile_95"] >= res["percentile_5"]


# ═══════════════════════════════════════════════════════════════════════
#  Portfolio Risk & Correlation Tests
# ═══════════════════════════════════════════════════════════════════════

def test_portfolio_risk_and_correlation():
    bars_dict = {
        "RELIANCE": [{"close": 2000 + i * 5} for i in range(30)],
        "TCS": [{"close": 3500 + i * 3} for i in range(30)],
        "INFY": [{"close": 1500 + i * 2} for i in range(30)],
    }
    corr = compute_correlation_matrix(bars_dict)
    assert not corr.empty
    assert "RELIANCE" in corr.columns
    assert np.isclose(corr.loc["RELIANCE", "RELIANCE"], 1.0)

    positions = [
        {"Ticker": "RELIANCE", "Current": 2150.0, "Qty": 100},
        {"Ticker": "TCS", "Current": 3590.0, "Qty": 50},
    ]
    metrics = compute_portfolio_metrics(bars_dict, positions)
    assert "herfindahl_index" in metrics
    assert "max_concentration_pct" in metrics
    assert "sector_exposure" in metrics
    assert "Energy" in metrics["sector_exposure"] or "IT" in metrics["sector_exposure"]


# ═══════════════════════════════════════════════════════════════════════
#  VWAP Strategy Tests
# ═══════════════════════════════════════════════════════════════════════

def test_vwap_strategy():
    strat = VWAPStrategy(tickers=["TCS"], min_bars=5, volume_factor=1.2)
    now = datetime.now(timezone.utc)

    # Feed bars below VWAP with normal volume
    signals = []
    for i in range(10):
        bar = Bar(
            ticker="TCS",
            timestamp=now + timedelta(minutes=i),
            open=3000.0,
            high=3010.0,
            low=2990.0,
            close=3000.0,
            volume=1000.0,
        )
        sig = strat.on_bar(bar)
        if sig:
            signals.append(sig)

    # Bullish crossover with massive volume surge
    breakout_bar = Bar(
        ticker="TCS",
        timestamp=now + timedelta(minutes=11),
        open=3000.0,
        high=3050.0,
        low=3000.0,
        close=3045.0,
        volume=5000.0,  # 5x volume
    )
    sig = strat.on_bar(breakout_bar)
    assert sig is not None
    assert sig.signal_type == SignalType.LONG_ENTRY
    assert sig.strength >= 0.6


# ═══════════════════════════════════════════════════════════════════════
#  Options Signal Strategy Tests
# ═══════════════════════════════════════════════════════════════════════

def test_options_signal_strategy_without_data():
    strat = OptionsSignalStrategy(tickers=["NIFTY"])
    bar = Bar(
        ticker="NIFTY",
        timestamp=datetime.now(timezone.utc),
        open=24000.0, high=24050.0, low=23950.0, close=24000.0, volume=100.0
    )
    # With no options data in state file, should gracefully return None
    sig = strat.on_bar(bar)
    assert sig is None


# ═══════════════════════════════════════════════════════════════════════
#  Candle Maker & Order Flow Volume Profile Tests
# ═══════════════════════════════════════════════════════════════════════

def test_live_candle_maker_aggregation():
    from analytics.candle_maker import LiveCandleMaker, LiveTick
    import pandas as pd

    maker = LiveCandleMaker()
    base_time = datetime(2026, 7, 3, 9, 15, 10, tzinfo=timezone.utc)

    # Ingest 3 ticks in the same 60s window
    maker.ingest_tick(LiveTick(ticker="RELIANCE", price=2950.0, volume=100.0, timestamp=base_time))
    maker.ingest_tick(LiveTick(ticker="RELIANCE", price=2955.0, volume=50.0, timestamp=base_time + timedelta(seconds=15)))
    maker.ingest_tick(LiveTick(ticker="RELIANCE", price=2948.0, volume=80.0, timestamp=base_time + timedelta(seconds=30)))

    forming = maker.get_forming_candle("RELIANCE")
    assert forming is not None
    assert forming.open == 2950.0
    assert forming.high == 2955.0
    assert forming.low == 2948.0
    assert forming.close == 2948.0
    assert forming.volume == 230.0
    assert forming.tick_count == 3

    # Tick crossing into next minute completes the previous bar
    completed = maker.ingest_tick(
        LiveTick(ticker="RELIANCE", price=2960.0, volume=120.0, timestamp=base_time + timedelta(seconds=65))
    )
    assert completed is not None
    assert completed["high"] == 2955.0
    assert completed["low"] == 2948.0
    assert completed["volume"] == 230.0


def test_volume_profile_poc_vah_val():
    from analytics.candle_maker import compute_volume_profile
    import pandas as pd

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-03", periods=30, freq="1min"),
        "open": [100.0 + i for i in range(30)],
        "high": [102.0 + i for i in range(30)],
        "low": [99.0 + i for i in range(30)],
        "close": [101.0 + i for i in range(30)],
        "volume": [100.0 if i != 15 else 5000.0 for i in range(30)],  # Heavy volume cluster at bar 15
    })

    vp = compute_volume_profile(df, num_bins=20)
    assert vp["poc"] > 0.0
    assert vp["vah"] >= vp["val"]
    assert vp["total_volume"] > 0.0


def test_heikin_ashi_and_anchored_vwap():
    from analytics.candle_maker import compute_heikin_ashi, compute_anchored_vwap_bands
    import pandas as pd

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-03", periods=20, freq="1min"),
        "open": np.linspace(100, 110, 20),
        "high": np.linspace(102, 112, 20),
        "low": np.linspace(99, 109, 20),
        "close": np.linspace(101, 111, 20),
        "volume": np.ones(20) * 1000.0,
    })

    ha = compute_heikin_ashi(df)
    assert len(ha) == len(df)
    assert "close" in ha.columns

    vwap, u1, l1, u2, l2 = compute_anchored_vwap_bands(df)
    assert len(vwap) == len(df)
    assert (u1 >= vwap).all()
    assert (l1 <= vwap).all()
    assert (u2 >= u1).all()
    assert (l2 <= l1).all()


def test_corporate_actions_upcoming_dividends():
    from data_engine.corporate_actions import CorporateActionsPipeline
    pipeline = CorporateActionsPipeline()
    pipeline._build_upcoming_schedule()

    upcoming = pipeline.dividend_results.get("upcoming", [])
    assert len(upcoming) > 0

    item = upcoming[0]
    assert "Ticker" in item
    assert "Last Date to Buy" in item
    assert "Ex-Dividend Date" in item
    assert "Hold Until" in item
    assert "When to Exit" in item
    assert "09:15 AM" in item["Hold Until"]


def test_trade_recommender_generation():
    from analytics.trade_recommender import TradeRecommender, TradeType
    recommender = TradeRecommender()

    # Test baseline fallback generation
    sugs = recommender.generate_recommendations(
        bars_dict={},
        options_chains={},
        regime_info={"regime": "TRENDING_UP"},
        prev_closes={},
    )
    assert len(sugs) >= 3

    # Check structure of suggestion
    first = sugs[0]
    assert "id" in first
    assert "ticker" in first
    assert "trade_type" in first
    assert "instrument_name" in first
    assert "action_label" in first
    assert "entry_price" in first
    assert "target_1" in first
    assert "target_2" in first
    assert "stop_loss" in first
    assert "risk_reward" in first
    assert first["confidence_pct"] >= 70
    assert "confluence_factors" in first
    assert len(first["confluence_factors"]) > 0

    # Test with mock options chain
    mock_chains = {
        "NIFTY 50": {
            "spot_price": 24500.0,
            "atm_strike": 24500.0,
            "pcr_oi": 1.35,
            "max_pain_strike": 24400.0,
            "iv_rank": 45.0,
            "strikes": [
                {"strike": 24500.0, "call_ltp": 180.0, "call_delta": 0.52, "call_gamma": 0.0018, "call_theta": -12.0, "call_vega": 18.0, "put_ltp": 150.0, "put_delta": -0.48},
            ]
        }
    }
    sugs2 = recommender.generate_recommendations(
        bars_dict={},
        options_chains=mock_chains,
        regime_info={"regime": "TRENDING_UP"},
        prev_closes={},
    )
    assert any(s["trade_type"] == TradeType.BUY_CALL.value for s in sugs2)



