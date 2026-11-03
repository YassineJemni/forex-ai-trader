from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.research.mission11_portfolio import (
    PortfolioLimits, build_allocations, coordinate_signals, currency_components,
    simulate_portfolio,
)


def row(strategy="ema_crossover", pair="EURUSD", direction="BUY", *,
        at="2026-01-01T00:00:00Z", exit_at="2026-01-01T01:00:00Z", pnl_r=1.0,
        regime="trending"):
    entry = pd.Timestamp(at)
    exit_time = pd.Timestamp(exit_at)
    pip = 0.0001 if not pair.endswith("JPY") else 0.01
    stop_distance = 10.0
    sign = 1 if direction == "BUY" else -1
    entry_price = 1.1 if not pair.endswith("JPY") else 150.0
    stop = entry_price - sign * stop_distance * pip
    target = entry_price + sign * 15 * pip
    result = {"strategy_id": strategy, "pair": pair, "timeframe": "M15",
            "decision_timestamp": entry - pd.Timedelta(minutes=15), "entry_timestamp": entry,
            "exit_timestamp": exit_time, "direction": direction, "entry_price": entry_price,
            "stop_loss": stop, "take_profit": target, "exit_price": entry_price + sign * pnl_r * stop_distance * pip,
            "exit_reason": "take_profit" if pnl_r > 0 else "stop_loss",
            "gross_pnl_pips": pnl_r * stop_distance, "stop_distance_pips": stop_distance,
            "risk_reward": 1.5, "regime": regime, "signal_evidence": "test signal"}
    result.update({"position_units": 100_000.0, "risk_amount": 50.0})
    return result


def equal_allocation(*strategies):
    return {"equal_risk": {s: 1.0 for s in strategies}}


def prepare(rows, correlations=None):
    return coordinate_signals(pd.DataFrame(rows), correlations or {})


def test_currency_exposure_uses_actual_base_quote_direction():
    assert currency_components("EURUSD", "BUY") == {"EUR": 1, "USD": -1}
    assert currency_components("USDCHF", "SELL") == {"USD": -1, "CHF": 1}
    assert currency_components("USDJPYm", "BUY") == {"USD": 1, "JPY": -1}


def test_coordinator_collapses_duplicates_and_distinguishes_correlation():
    candidates, rejected, counts = prepare([
        row("ema_crossover"), row("trend_pullback_momentum"), row("structure_bos")
    ], {("ema_crossover", "trend_pullback_momentum"): 0.91})
    assert not rejected
    assert len(candidates) == 1
    assert candidates[0]["signal_ids"] == ["ema_crossover", "structure_bos", "trend_pullback_momentum"]
    assert candidates[0]["correlated_confirmation_ids"] == ["trend_pullback_momentum"]
    assert candidates[0]["unknown_correlation_confirmation_ids"] == ["structure_bos"]
    assert counts["duplicate_signals_collapsed"] == 2
    assert counts["independent_confirmations"] == 0


def test_coordinator_rejects_opposing_same_bar_signals():
    candidates, rejected, counts = prepare([row("ema_crossover", direction="BUY"),
                                            row("rsi14_recross", direction="SELL")])
    assert candidates == []
    assert rejected[0]["reason"] == "simultaneous_opposing_strategy_signals"
    assert {item["direction"] for item in rejected[0]["signal_details"]} == {"BUY", "SELL"}
    assert counts["signal_conflict"] == 1


def test_separate_pairs_run_simultaneously_but_shared_usd_exposure_is_capped():
    when = "2026-01-01T00:00:00Z"
    opportunities, conflicts, _ = prepare([
        row("ema_crossover", "EURUSD", "BUY", at=when),
        row("donchian_breakout", "GBPUSD", "BUY", at=when),
        row("structure_bos", "USDCAD", "BUY", at=when),
    ])
    result = simulate_portfolio(opportunities, conflicts, portfolio="equal_risk",
                                allocation=equal_allocation(*["ema_crossover", "donchian_breakout", "structure_bos"]),
                                correlations={}, limits=PortfolioLimits(max_currency_open_risk=.01),
                                costs_pips=0, split="test")
    assert len(result["trades"]) == 2
    assert result["rejected_reason_counts"]["currency_exposure_limit_USD"] == 1
    assert result["final_open_positions"] == 0


def test_maximum_simultaneous_positions_limit_is_shared_across_pairs():
    when = "2026-01-01T00:00:00Z"
    opportunities, conflicts, _ = prepare([
        row("ema_crossover", "EURUSD", "BUY", at=when),
        row("donchian_breakout", "USDJPY", "BUY", at=when),
    ])
    result = simulate_portfolio(opportunities, conflicts, portfolio="equal_risk",
                                allocation=equal_allocation("ema_crossover", "donchian_breakout"),
                                correlations={}, limits=PortfolioLimits(max_positions=1,
                                                                         max_currency_open_risk=.05),
                                costs_pips=0, split="test")
    assert len(result["trades"]) == 1
    assert result["rejected_reason_counts"]["maximum_simultaneous_positions"] == 1


def test_costs_are_applied_in_r_and_outcomes_realize_only_at_exit():
    opportunities, conflicts, _ = prepare([row(pnl_r=2.0)])
    result = simulate_portfolio(opportunities, conflicts, portfolio="equal_risk",
                                allocation=equal_allocation("ema_crossover"), correlations={},
                                limits=PortfolioLimits(), costs_pips=.5, split="test")
    trade = result["trades"][0]
    assert trade["net_r"] == pytest.approx(1.95)
    assert result["metrics"]["ending_equity_net"] == pytest.approx(10_097.5)
    assert trade["simulated_position_units"] == pytest.approx(100_000.0)
    assert trade["risk_cash"] == pytest.approx(50.0)
    entry_points = [point for point in result["equity_curve"] if point["timestamp_utc"].startswith("2026-01-01T00")]
    assert entry_points[0]["equity_net"] == 10_000


def test_same_pair_duplicate_and_opposite_active_position_are_rejected():
    rows = [row(at="2026-01-01T00:00:00Z", exit_at="2026-01-01T03:00:00Z"),
            row("structure_bos", at="2026-01-01T01:00:00Z", exit_at="2026-01-01T04:00:00Z"),
            row("rsi14_recross", direction="SELL", at="2026-01-01T02:00:00Z")]
    opportunities, conflicts, _ = prepare(rows)
    result = simulate_portfolio(opportunities, conflicts, portfolio="equal_risk",
                                allocation=equal_allocation("ema_crossover", "structure_bos", "rsi14_recross"),
                                correlations={}, limits=PortfolioLimits(max_currency_open_risk=.05),
                                costs_pips=0, split="test")
    assert len(result["trades"]) == 1
    assert result["rejected_reason_counts"]["duplicate_active_pair_opportunity"] == 1
    assert result["rejected_reason_counts"]["conflicting_active_pair_position"] == 1


def test_daily_loss_and_drawdown_limits_stop_new_entries():
    rows = [row(pnl_r=-1, at="2026-01-01T00:00:00Z", exit_at="2026-01-01T01:00:00Z"),
            row("structure_bos", pnl_r=1, at="2026-01-01T02:00:00Z", exit_at="2026-01-01T03:00:00Z")]
    opportunities, conflicts, _ = prepare(rows)
    limits = PortfolioLimits(max_daily_loss=.006, max_drawdown=.006)
    result = simulate_portfolio(opportunities, conflicts, portfolio="equal_risk",
                                allocation=equal_allocation("ema_crossover", "structure_bos"),
                                correlations={}, limits=limits, costs_pips=0, split="test")
    assert len(result["trades"]) == 1
    assert result["rejected_reason_counts"].get("maximum_daily_loss_limit", 0) + result["rejected_reason_counts"].get("maximum_drawdown_limit", 0) == 1


def test_allocation_maps_are_validation_derived_and_keep_weak_strategies_available():
    mission9 = {"metrics": [
        {"strategy_id": "ema_crossover", "split": "validation",
         "scenario": "spread_2_commission_1_slippage_0.5_pips_rt", "trades": 10, "total_r": 2.0},
        {"strategy_id": "structure_bos", "split": "validation",
         "scenario": "spread_2_commission_1_slippage_0.5_pips_rt", "trades": 10, "total_r": -3.0},
        {"strategy_id": "ema_crossover", "split": "test",
         "scenario": "spread_2_commission_1_slippage_0.5_pips_rt", "trades": 100, "total_r": 99.0},
    ], "market_regime": [], "daily_return_correlations": []}
    maps = build_allocations(mission9, {})
    assert maps["validation_evidence"]["test_used"] is False
    assert maps["performance_weighted"]["ema_crossover"] > maps["performance_weighted"]["structure_bos"]
    assert maps["performance_weighted"]["structure_bos"] > 0
    assert set(maps["equal_risk"]) >= {"structure_bos", "ema_crossover"}


def test_regime_specialization_uses_validation_regime_rows_only():
    mission9 = {"metrics": [], "daily_return_correlations": [], "market_regime": [
        {"strategy_id": "ema_crossover", "regime": "trending", "split": "validation",
         "scenario": "spread_2_commission_1_slippage_0.5_pips_rt", "trades": 10, "total_r": 4.0},
        {"strategy_id": "ema_crossover", "regime": "trending", "split": "test",
         "scenario": "spread_2_commission_1_slippage_0.5_pips_rt", "trades": 100, "total_r": 90.0},
    ]}
    maps = build_allocations(mission9, {})
    assert maps["regime_specialized"]["ema_crossover"]["trending"] > maps["regime_specialized"]["ema_crossover"]["ranging"]


def test_simulation_is_reproducible_for_identical_events_and_limits():
    opportunities, conflicts, _ = prepare([row(), row("structure_bos", pair="AUDUSD")])
    kwargs = {"portfolio": "equal_risk", "allocation": equal_allocation("ema_crossover", "structure_bos"),
              "correlations": {}, "limits": PortfolioLimits(), "costs_pips": .5, "split": "test"}
    first = simulate_portfolio(opportunities, conflicts, **kwargs)
    second = simulate_portfolio(opportunities, conflicts, **kwargs)
    assert first["metrics"] == second["metrics"]
    assert first["trades"] == second["trades"]


def test_limits_reject_invalid_risk_settings():
    with pytest.raises(ValueError):
        PortfolioLimits(risk_per_trade=.03)
