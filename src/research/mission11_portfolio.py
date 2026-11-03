"""Mission 11: read-only multi-strategy portfolio simulation.

This engine reuses Mission 8's saved next-bar entries and intrabar-resolved
trade outcomes. It never regenerates signals or modifies market datasets.
Allocation maps are fitted from Mission 9 validation evidence only; test-period
results are reported as a locked descriptive holdout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mission9_cost_research import _assign_splits, _research_split_bounds

ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = (
    "ema_crossover", "trend_pullback_momentum", "donchian_breakout", "breakout_retest",
    "rsi14_recross", "bollinger_reentry", "structure_bos", "structure_choch",
    "structure_hh_hl", "liquidity_sweep", "fair_value_gap", "order_block_hypothesis",
    "displacement_rejection", "engulfing_at_level",
)
PORTFOLIOS = ("equal_risk", "regime_specialized", "performance_weighted",
              "correlation_aware", "conservative")
STRESS_COSTS = {"spread_round_trip_pips": 2.0, "commission_round_trip_pips": 1.0,
                "slippage_round_trip_pips": 0.5, "source": "Mission 9 declared stress scenario; all components are assumptions"}


@dataclass(frozen=True)
class PortfolioLimits:
    initial_balance: float = 10_000.0
    risk_per_trade: float = 0.005
    max_total_open_risk: float = 0.02
    max_pair_open_risk: float = 0.0075
    max_currency_open_risk: float = 0.01
    max_positions: int = 8
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.10

    def __post_init__(self) -> None:
        if self.initial_balance <= 0 or not 0 < self.risk_per_trade <= 0.02:
            raise ValueError("initial balance must be positive and risk_per_trade in (0, .02]")
        for name in ("max_total_open_risk", "max_pair_open_risk", "max_currency_open_risk",
                     "max_daily_loss", "max_drawdown"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least one")


def currency_components(pair: str, direction: str | int) -> dict[str, int]:
    """Return signed base/quote currency exposure for BUY/SELL pair direction."""
    compact = "".join(ch for ch in pair.upper() if ch.isalpha())
    if len(compact) < 6:
        raise ValueError(f"Invalid currency pair: {pair!r}")
    if isinstance(direction, str):
        side = 1 if direction.upper() == "BUY" else -1 if direction.upper() == "SELL" else 0
    else:
        side = 1 if direction > 0 else -1 if direction < 0 else 0
    if not side:
        raise ValueError(f"Invalid direction: {direction!r}")
    return {compact[:3]: side, compact[3:6]: -side}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _correlation_map(mission9: dict[str, Any]) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
    rows = [r for r in mission9.get("daily_return_correlations", [])
            if r.get("split") == "validation" and
            r.get("scenario") == "spread_2_commission_1_slippage_0.5_pips_rt"]
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("daily_exit_return_correlation")
        if value is not None and np.isfinite(float(value)):
            values[_pair_key(row["strategy_a"], row["strategy_b"])].append(float(value))
    corr = {key: float(np.median(items)) for key, items in values.items()}
    strategies = sorted({name for pair in corr for name in pair})
    matrix = []
    for a in strategies:
        matrix.append({"strategy_id": a, **{b: (1.0 if a == b else corr.get(_pair_key(a, b))) for b in strategies}})
    return corr, matrix


def _risk_multiplier(expectancy: float | None) -> float:
    """Bound validation-derived risk between 35% and 100%; never removes a strategy."""
    if expectancy is None or not np.isfinite(expectancy):
        return 0.35
    score = float(np.clip((expectancy + 0.25) / 0.50, 0.0, 1.0))
    return 0.35 + 0.65 * score


def build_allocations(mission9: dict[str, Any], correlations: dict[tuple[str, str], float]) -> dict[str, Any]:
    """Fit all non-equal allocation maps using validation split rows only."""
    stress = "spread_2_commission_1_slippage_0.5_pips_rt"
    metrics = [r for r in mission9.get("metrics", [])
               if r.get("split") == "validation" and r.get("scenario") == stress]
    per_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        per_strategy[row["strategy_id"]].append(row)
    expectancy = {}
    for strategy in STRATEGIES:
        rows = per_strategy.get(strategy, [])
        count = sum(int(x.get("trades", 0)) for x in rows)
        total = sum(float(x.get("total_r", 0.0)) for x in rows)
        expectancy[strategy] = total / count if count else None
    performance = {s: _risk_multiplier(expectancy[s]) for s in STRATEGIES}

    regime_rows = [r for r in mission9.get("market_regime", [])
                   if r.get("split") == "validation" and r.get("scenario") == stress]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in regime_rows:
        grouped[(row["strategy_id"], row["regime"])].append(row)
    regime = {}
    for strategy in STRATEGIES:
        regime[strategy] = {}
        for market_regime in ("trending", "ranging", "high_volatility", "low_volatility", "transitional_uncertain"):
            rows = grouped.get((strategy, market_regime), [])
            n = sum(int(x.get("trades", 0)) for x in rows)
            total = sum(float(x.get("total_r", 0.0)) for x in rows)
            regime[strategy][market_regime] = _risk_multiplier(total / n if n else None)

    mean_positive_corr = {}
    for strategy in STRATEGIES:
        peers = [max(0.0, value) for pair, value in correlations.items() if strategy in pair]
        mean_positive_corr[strategy] = float(np.mean(peers)) if peers else 0.0
    correlation_aware = {s: max(0.25, performance[s] / (1.0 + mean_positive_corr[s])) for s in STRATEGIES}
    conservative = {s: (0.25 if expectancy[s] is None or expectancy[s] <= 0 else 0.50) for s in STRATEGIES}
    return {"equal_risk": {s: 1.0 for s in STRATEGIES},
            "regime_specialized": regime,
            "performance_weighted": performance,
            "correlation_aware": correlation_aware,
            "conservative": conservative,
            "validation_evidence": {"strategy_expectancy_r": expectancy,
                                    "mean_positive_validation_correlation": mean_positive_corr,
                                    "source_split": "validation", "test_used": False}}


def _factor(allocation: dict[str, Any], portfolio: str, strategy: str, regime: str) -> float:
    if portfolio == "regime_specialized":
        return float(allocation[portfolio].get(strategy, {}).get(regime, 0.35))
    return float(allocation[portfolio].get(strategy, 0.35))


def coordinate_signals(trades: pd.DataFrame, correlations: dict[tuple[str, str], float],
                       *, high_correlation_threshold: float = 0.80) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Merge same-bar duplicate signals, retain confirmation/conflict provenance."""
    needed = ["strategy_id", "pair", "timeframe", "decision_timestamp", "entry_timestamp", "exit_timestamp",
              "direction", "entry_price", "stop_loss", "take_profit", "exit_price", "exit_reason",
              "gross_pnl_pips", "stop_distance_pips", "position_units", "risk_amount",
              "regime", "signal_evidence"]
    frame = trades[needed].copy()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts = Counter()
    # Exact pair/timeframe/entry bars identify one market opportunity; different
    # directional signals on that bar cancel rather than vote by majority.
    keys = ["pair", "timeframe", "entry_timestamp"]
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        dirs = set(group.direction.astype(str).str.upper())
        signal_ids = sorted(group.strategy_id.astype(str).unique())
        if len(dirs) > 1:
            counts["signal_conflict"] += 1
            rejected.append({"pair": key[0], "timeframe": key[1], "entry_timestamp": key[2],
                             "strategy_ids": signal_ids,
                             "signal_details": [{"strategy_id": r.strategy_id, "direction": r.direction,
                                                 "signal_evidence": r.signal_evidence}
                                                for r in group.itertuples(index=False)],
                             "reason": "simultaneous_opposing_strategy_signals"})
            continue
        ordered = group.sort_values("strategy_id", kind="stable")
        members = ordered.to_dict("records")
        origin = members[0]
        supporting = [m["strategy_id"] for m in members if m["strategy_id"] != origin["strategy_id"]]
        correlated, independent, unknown = [], [], []
        classification_basis = [origin["strategy_id"]]
        for other in supporting:
            values = [correlations.get(_pair_key(known, other)) for known in classification_basis]
            if any(value is not None and abs(value) >= high_correlation_threshold for value in values):
                correlated.append(other)
            elif any(value is None for value in values):
                unknown.append(other)
            else:
                independent.append(other)
            classification_basis.append(other)
        counts["coordinated_opportunities"] += 1
        counts["duplicate_signals_collapsed"] += max(0, len(members) - 1)
        counts["independent_confirmations"] += len(independent)
        counts["highly_correlated_confirmations"] += len(correlated)
        counts["unknown_correlation_confirmations"] += len(unknown)
        candidates.append({"pair": key[0], "timeframe": key[1], "entry_timestamp": key[2],
                          "direction": next(iter(dirs)), "members": members,
                          "signal_ids": signal_ids, "supporting_signal_ids": supporting,
                          "correlated_confirmation_ids": correlated,
                          "independent_confirmation_ids": independent,
                          "unknown_correlation_confirmation_ids": unknown})
    candidates.sort(key=lambda x: (x["entry_timestamp"], x["pair"], x["timeframe"], x["direction"], x["signal_ids"]))
    return candidates, rejected, dict(counts)


def _candidate_initiator(candidate: dict[str, Any], allocation: dict[str, Any], portfolio: str) -> dict[str, Any]:
    return max(candidate["members"], key=lambda row: (_factor(allocation, portfolio, row["strategy_id"], row["regime"]),
                                                        row["strategy_id"]))


def _close_due(active: list[dict[str, Any]], current: pd.Timestamp, state: dict[str, Any],
               curves: list[dict[str, Any]], output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    due = [position for position in active if position["exit_time"] <= current]
    remains = [position for position in active if position["exit_time"] > current]
    # Realize simultaneous exits as one timestamped portfolio event so the
    # result does not depend on arbitrary row ordering within the same bar.
    by_time: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    for position in due:
        by_time[position["exit_time"]].append(position)
    for exit_time in sorted(by_time):
        closed = by_time[exit_time]
        pnl_net = sum(p["pnl_net_cash"] for p in closed)
        pnl_gross = sum(p["pnl_gross_cash"] for p in closed)
        state["equity_net"] += pnl_net
        state["equity_gross"] += pnl_gross
        state["realized_by_day"][exit_time.date()] += pnl_net
        state["closed"].extend(closed)
        state["peak"] = max(state["peak"], state["equity_net"])
        curves.append({"timestamp_utc": exit_time.isoformat(),
                       "equity_net": state["equity_net"], "equity_gross": state["equity_gross"],
                       "drawdown_net": state["peak"] - state["equity_net"],
                       "open_positions": len(remains),
                       "open_risk_cash": sum(p["risk_cash"] for p in remains)})
    return remains


def simulate_portfolio(candidates: list[dict[str, Any]], conflicts: list[dict[str, Any]], *,
                       portfolio: str, allocation: dict[str, Any], correlations: dict[tuple[str, str], float],
                       limits: PortfolioLimits, costs_pips: float, split: str,
                       collect_rejections: bool = False) -> dict[str, Any]:
    """Chronological event simulation with shared pair/currency/risk constraints."""
    if portfolio not in PORTFOLIOS or costs_pips < 0:
        raise ValueError("unknown portfolio or negative transaction costs")
    active: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejects = Counter()
    rejected_sample: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    state = {"equity_net": limits.initial_balance, "equity_gross": limits.initial_balance,
             "peak": limits.initial_balance, "realized_by_day": defaultdict(float),
             "day_start_equity": {}, "closed": []}
    max_drawdown_cash = 0.0
    for candidate in candidates:
        when = pd.Timestamp(candidate["entry_timestamp"])
        active = _close_due(active, when, state, curve, accepted)
        equity = state["equity_net"]
        current_day = when.date()
        state["day_start_equity"].setdefault(current_day, equity - state["realized_by_day"][current_day])
        if equity <= 0:
            reason = "portfolio_equity_depleted"
        elif len(active) >= limits.max_positions:
            reason = "maximum_simultaneous_positions"
        else:
            reason = ""
        initiator = _candidate_initiator(candidate, allocation, portfolio)
        factor = float(np.clip(_factor(allocation, portfolio, initiator["strategy_id"], initiator["regime"]), 0.0, 1.0))
        risk_cash = equity * limits.risk_per_trade * factor
        stop_distance = max(float(initiator["stop_distance_pips"]), 1e-12)
        worst_case_loss_cash = risk_cash * (1.0 + costs_pips / stop_distance)
        reserved_loss = sum(p["worst_case_loss_cash"] for p in active) + worst_case_loss_cash
        drawdown_to_date = max(0.0, state["peak"] - equity)
        if not reason and drawdown_to_date + reserved_loss > state["peak"] * limits.max_drawdown + 1e-9:
            reason = "maximum_drawdown_limit"
        daily_loss_used = max(0.0, -state["realized_by_day"][current_day])
        if not reason and daily_loss_used + reserved_loss > state["day_start_equity"][current_day] * limits.max_daily_loss + 1e-9:
            reason = "maximum_daily_loss_limit"
        same_pair = [p for p in active if p["pair"] == candidate["pair"]]
        if not reason and any(p["direction"] == candidate["direction"] for p in same_pair):
            reason = "duplicate_active_pair_opportunity"
        elif not reason and same_pair:
            reason = "conflicting_active_pair_position"
        open_risk = sum(p["risk_cash"] for p in active)
        if not reason and open_risk + risk_cash > equity * limits.max_total_open_risk + 1e-9:
            reason = "total_open_risk_limit"
        pair_risk = sum(p["risk_cash"] for p in active if p["pair"] == candidate["pair"])
        if not reason and pair_risk + risk_cash > equity * limits.max_pair_open_risk + 1e-9:
            reason = "pair_exposure_limit"
        components = currency_components(candidate["pair"], candidate["direction"])
        for currency in components:
            currency_risk = sum(p["risk_cash"] for p in active if currency in p["currency_components"])
            if not reason and currency_risk + risk_cash > equity * limits.max_currency_open_risk + 1e-9:
                reason = f"currency_exposure_limit_{currency}"
        if reason:
            rejects[reason] += 1
            if collect_rejections and len(rejected_sample) < 5000:
                rejected_sample.append({"portfolio": portfolio, "split": split, "pair": candidate["pair"],
                                        "timeframe": candidate["timeframe"], "entry_timestamp": candidate["entry_timestamp"],
                                        "direction": candidate["direction"], "strategy_ids": candidate["signal_ids"],
                                        "reason": reason})
            continue
        gross_r = float(initiator["gross_pnl_pips"]) / max(float(initiator["stop_distance_pips"]), 1e-12)
        net_r = gross_r - costs_pips / max(float(initiator["stop_distance_pips"]), 1e-12)
        pnl_gross = risk_cash * gross_r
        pnl_net = risk_cash * net_r
        members = candidate["members"]
        event = {"portfolio": portfolio, "split": split, "pair": candidate["pair"],
                 "timeframe": candidate["timeframe"], "entry_timestamp": candidate["entry_timestamp"],
                 "decision_timestamp": initiator["decision_timestamp"],
                 "exit_timestamp": initiator["exit_timestamp"], "direction": candidate["direction"],
                 "origin_strategy": initiator["strategy_id"], "strategy_ids": candidate["signal_ids"],
                 "member_signal_evidence": [{"strategy_id": member["strategy_id"],
                                             "direction": member["direction"],
                                             "decision_timestamp": str(member["decision_timestamp"]),
                                             "entry_price": member["entry_price"],
                                             "stop_loss": member["stop_loss"],
                                             "take_profit": member["take_profit"],
                                             "evidence": member["signal_evidence"],
                                             "regime": member["regime"]} for member in members],
                 "supporting_strategies": candidate["supporting_signal_ids"],
                 "independent_confirmations": candidate["independent_confirmation_ids"],
                 "highly_correlated_confirmations": candidate["correlated_confirmation_ids"],
                 "unknown_correlation_confirmations": candidate["unknown_correlation_confirmation_ids"],
                 "duplicate_signals_collapsed": len(members) - 1,
                 "regime": initiator["regime"], "signal_evidence": initiator["signal_evidence"],
                 "entry_price": initiator["entry_price"], "stop_loss": initiator["stop_loss"],
                 "take_profit": initiator["take_profit"], "exit_price": initiator["exit_price"],
                 "exit_reason": initiator["exit_reason"], "risk_cash": risk_cash,
                 "worst_case_loss_cash_reserved": worst_case_loss_cash,
                 "source_position_units": initiator.get("position_units"),
                 "simulated_position_units": (float(initiator["position_units"]) * risk_cash /
                                               float(initiator["risk_amount"]) if initiator.get("position_units") is not None
                                               and initiator.get("risk_amount") else None),
                 "allocation_multiplier": factor, "gross_r": gross_r, "net_r": net_r,
                 "gross_pnl_cash": pnl_gross, "net_pnl_cash": pnl_net,
                 "signal_validity": "saved_Mission8_entry_and_exit_validated",
                 "transaction_costs_round_trip_pips_assumed": costs_pips,
                 "risk_limit_decision": "accepted", "currency_components": components,
                 "attribution_split_per_strategy": 1.0 / len(members)}
        accepted.append(event)
        active.append({**event, "exit_time": pd.Timestamp(initiator["exit_timestamp"]),
                       "pnl_net_cash": pnl_net, "pnl_gross_cash": pnl_gross,
                       "worst_case_loss_cash": worst_case_loss_cash,
                       "currency_components": components})
        current_risk = sum(p["risk_cash"] for p in active)
        curve.append({"timestamp_utc": when.isoformat(), "equity_net": state["equity_net"],
                      "equity_gross": state["equity_gross"], "drawdown_net": max(0.0, state["peak"] - state["equity_net"]),
                      "open_positions": len(active), "open_risk_cash": current_risk})
    if active:
        final = max(p["exit_time"] for p in active) + pd.Timedelta(nanoseconds=1)
        active = _close_due(active, final, state, curve, accepted)
    for item in conflicts:
        rejects["simultaneous_opposing_strategy_signals"] += 1
        if collect_rejections and len(rejected_sample) < 5000:
            rejected_sample.append({"portfolio": portfolio, "split": split, **item})
    curve_frame = pd.DataFrame(curve)
    accepted_frame = pd.DataFrame(accepted)
    final_equity = state["equity_net"]
    gross_final = state["equity_gross"]
    if not accepted_frame.empty:
        exits = accepted_frame.sort_values("exit_timestamp", kind="stable")
        net_values = exits.net_r.to_numpy(float)
        wins = net_values[net_values > 0]
        losses = net_values[net_values < 0]
        loss_streak = max_streak = 0
        for val in net_values:
            loss_streak = loss_streak + 1 if val < 0 else 0
            max_streak = max(max_streak, loss_streak)
        daily = exits.assign(exit_date=pd.to_datetime(exits.exit_timestamp, utc=True).dt.floor("D"))
        daily_pnl = daily.groupby("exit_date").net_pnl_cash.sum()
        if len(daily_pnl):
            calendar = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D", tz="UTC")
            daily_equity = (limits.initial_balance + daily_pnl.reindex(calendar, fill_value=0).cumsum())
            returns = daily_equity.pct_change().dropna().to_numpy(float)
        else:
            returns = np.array([], dtype=float)
        downside = returns[returns < 0]
        sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(365)) if len(returns) > 1 and returns.std(ddof=1) else None
        sortino = float(returns.mean() / downside.std(ddof=1) * math.sqrt(365)) if len(downside) > 1 and downside.std(ddof=1) else None
        risk_total = float(accepted_frame.risk_cash.sum())
        gross_total = float(accepted_frame.gross_pnl_cash.sum())
        net_total = float(accepted_frame.net_pnl_cash.sum())
        cash_pnl = accepted_frame.net_pnl_cash.to_numpy(float)
        cash_gains = float(cash_pnl[cash_pnl > 0].sum())
        cash_losses = float(-cash_pnl[cash_pnl < 0].sum())
        profit_factor = cash_gains / cash_losses if cash_losses else ("infinite" if cash_gains else None)
        values = pd.concat([pd.Series([limits.initial_balance]), daily_equity]) if len(daily_pnl) else pd.Series([limits.initial_balance])
        dd = values.cummax() - values
        max_dd_cash = float(dd.max())
        trade_r_curve = exits.net_r.cumsum().to_numpy(float)
        max_dd_r = float((np.maximum.accumulate(np.r_[0.0, trade_r_curve])[:-1] - trade_r_curve).max()) if len(trade_r_curve) else 0.0
        if not curve_frame.empty:
            utilization = curve_frame.open_risk_cash / np.maximum(curve_frame.equity_net * limits.max_total_open_risk, 1e-12)
            mean_exposure_util = float(utilization.mean())
            max_exposure_util = float(utilization.max())
        else:
            mean_exposure_util = max_exposure_util = 0.0
        metrics = {"trades": len(accepted_frame), "gross_return_fraction": gross_total / limits.initial_balance,
                   "net_return_fraction": net_total / limits.initial_balance,
                   "ending_equity_net": final_equity, "ending_equity_gross": gross_final,
                   "expectancy_r_net": net_total / risk_total if risk_total else None,
                   "expectancy_r_gross": gross_total / risk_total if risk_total else None,
                   "profit_factor_net": profit_factor,
                   "max_drawdown_cash": max_dd_cash,
                   "max_drawdown_fraction": max_dd_cash / max(limits.initial_balance, 1e-12),
                   "max_drawdown_r": max_dd_r,
                   "sharpe_daily_annualized_365": sharpe, "sortino_daily_annualized_365": sortino,
                   "win_rate_net": float((net_values > 0).mean()),
                   "average_win_r": float(wins.mean()) if len(wins) else None,
                   "average_loss_r": float(losses.mean()) if len(losses) else None,
                   "max_consecutive_losses": max_streak, "risk_adjusted_return_net_over_max_dd": net_total / max(max_dd_cash, 1e-12),
                   "mean_open_risk_utilization_of_total_limit": mean_exposure_util,
                   "max_open_risk_utilization_of_total_limit": max_exposure_util,
                   "costs_total_pips_are_assumptions": True}
    else:
        metrics = {"trades": 0, "gross_return_fraction": 0.0, "net_return_fraction": 0.0,
                   "ending_equity_net": limits.initial_balance, "ending_equity_gross": limits.initial_balance,
                   "expectancy_r_net": None, "expectancy_r_gross": None, "profit_factor_net": None,
                   "max_drawdown_cash": 0.0, "max_drawdown_fraction": 0.0,
                   "max_drawdown_r": 0.0,
                   "sharpe_daily_annualized_365": None, "sortino_daily_annualized_365": None,
                   "win_rate_net": None, "average_win_r": None, "average_loss_r": None,
                   "max_consecutive_losses": 0, "risk_adjusted_return_net_over_max_dd": 0.0,
                   "mean_open_risk_utilization_of_total_limit": 0.0,
                   "max_open_risk_utilization_of_total_limit": 0.0,
                   "costs_total_pips_are_assumptions": True}
    return {"metrics": metrics, "trades": accepted, "equity_curve": curve,
            "rejected_reason_counts": dict(rejects), "rejected_sample": rejected_sample,
            "final_open_positions": len(active)}


def _attribution(trades: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0, "gross_pnl_cash": 0.0,
                                                               "net_pnl_cash": 0.0, "risk_cash": 0.0})
    for trade in trades:
        if key == "strategy_id":
            contributors = trade["strategy_ids"]
            share = trade["attribution_split_per_strategy"]
            labels = contributors
        else:
            labels = [trade[key]]
            share = 1.0
        for label in labels:
            row = totals[str(label)]
            row["trades"] += share
            for field in ("gross_pnl_cash", "net_pnl_cash", "risk_cash"):
                row[field] += float(trade[field]) * share
    return [{key: label, **values,
             "expectancy_r_net": values["net_pnl_cash"] / values["risk_cash"] if values["risk_cash"] else None}
            for label, values in sorted(totals.items())]


def _currency_exposure(trades: list[dict[str, Any]], *, portfolio: str,
                       cost_scenario: str, split: str) -> list[dict[str, Any]]:
    events = []
    for trade in trades:
        for currency, sign in trade["currency_components"].items():
            events.append((pd.Timestamp(trade["exit_timestamp"]), 0, currency, sign, float(trade["risk_cash"])))
            events.append((pd.Timestamp(trade["entry_timestamp"]), 1, currency, sign, float(trade["risk_cash"])))
    current_gross: dict[str, float] = defaultdict(float)
    current_net: dict[str, float] = defaultdict(float)
    peaks: dict[str, dict[str, Any]] = defaultdict(lambda: {"max_concurrent_gross_risk_cash": 0.0,
                                                           "max_absolute_net_directional_risk_cash": 0.0,
                                                           "peak_net_directional_risk_cash": 0.0,
                                                           "peak_timestamp_utc": None,
                                                           "gross_risk_turnover_cash": 0.0})
    for when, kind, currency, sign, risk_cash in sorted(events, key=lambda x: (x[0], x[1], x[2])):
        if kind == 0:
            current_gross[currency] = max(0.0, current_gross[currency] - risk_cash)
            current_net[currency] -= sign * risk_cash
        else:
            current_gross[currency] += risk_cash
            current_net[currency] += sign * risk_cash
            peaks[currency]["gross_risk_turnover_cash"] += risk_cash
        peak = peaks[currency]
        if current_gross[currency] > peak["max_concurrent_gross_risk_cash"]:
            peak["max_concurrent_gross_risk_cash"] = current_gross[currency]
        if abs(current_net[currency]) > peak["max_absolute_net_directional_risk_cash"]:
            peak["max_absolute_net_directional_risk_cash"] = abs(current_net[currency])
            peak["peak_net_directional_risk_cash"] = current_net[currency]
            peak["peak_timestamp_utc"] = when.isoformat()
    return [{"portfolio": portfolio, "cost_scenario": cost_scenario, "split": split,
             "currency": currency, **values} for currency, values in sorted(peaks.items())]


def _independent_baselines(frame: pd.DataFrame, costs: dict[str, float], split: str, cost_scenario: str,
                           limits: PortfolioLimits) -> list[dict[str, Any]]:
    output = []
    cost_total = sum(costs.values())
    for strategy, group in frame.groupby("strategy_id", sort=False):
        ordered = group.sort_values(["exit_timestamp", "entry_timestamp"], kind="stable")
        gross_r = ordered.gross_pnl_pips.to_numpy(float) / np.maximum(ordered.stop_distance_pips.to_numpy(float), 1e-12)
        r = gross_r - cost_total / np.maximum(ordered.stop_distance_pips.to_numpy(float), 1e-12)
        cumulative_r = np.cumsum(r)
        drawdown_r = np.maximum.accumulate(np.r_[0.0, cumulative_r])[:-1] - cumulative_r
        wins, losses = r[r > 0], r[r < 0]
        pnl = r * limits.initial_balance * limits.risk_per_trade
        output.append({"portfolio": "independent_strategy", "strategy_id": strategy, "split": split,
                       "cost_scenario": cost_scenario,
                       "trades": len(group), "gross_return_fraction": float((ordered.gross_pnl_pips / ordered.stop_distance_pips * limits.risk_per_trade).sum()),
                       "net_return_fraction": float(pnl.sum() / limits.initial_balance),
                       "expectancy_r_net": float(r.mean()) if len(r) else None,
                       "max_drawdown_r": float(drawdown_r.max()) if len(drawdown_r) else 0.0,
                       "max_drawdown_fraction": None,
                       "profit_factor_net": float(wins.sum() / -losses.sum()) if len(losses) else ("infinite" if len(wins) else None),
                       "win_rate_net": float((r > 0).mean()) if len(r) else None,
                       "uses_portfolio_constraints": False,
                       "independent_trade_stream": "Unconstrained saved per-strategy events; normalized R drawdown is the comparable baseline metric.",
                       "cost_inputs_are_assumptions": True})
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _diversification_summary(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    table = pd.DataFrame(comparison)
    if table.empty:
        return output
    for (split, scenario), group in table.groupby(["split", "cost_scenario"], sort=False):
        equal = group[group.portfolio == "equal_risk"]
        individual = group[group.portfolio == "independent_strategy"]
        if equal.empty or individual.empty:
            continue
        eq = equal.iloc[0]
        median_trades = float(individual.trades.median())
        median_dd = float(individual.max_drawdown_r.median())
        output.append({"split": split, "cost_scenario": scenario,
                       "equal_risk_portfolio_trades": int(eq.trades),
                       "median_individual_strategy_trades": median_trades,
                       "trade_count_multiple_vs_median_individual": float(eq.trades / median_trades) if median_trades else None,
                       "equal_risk_portfolio_max_drawdown_r": float(eq.max_drawdown_r),
                       "median_individual_strategy_max_drawdown_r": median_dd,
                       "drawdown_below_median_individual": bool(eq.max_drawdown_r < median_dd),
                       "interpretation": "Descriptive R-scale comparison; individual baselines have no shared portfolio limits and avoid misleading fixed-capital compounding."})
    return output


def _latest(directory: Path, pattern: str) -> Path:
    options = list(directory.glob(pattern))
    if not options:
        raise FileNotFoundError(f"No {pattern} under {directory}")
    return max(options, key=lambda x: x.stat().st_mtime)


def run_mission11(mission8_path: Path, mission9_path: Path, trade_path: Path, output_dir: Path,
                  *, limits: PortfolioLimits | None = None, stress_costs: dict[str, float] | None = None) -> dict[str, Any]:
    limits = limits or PortfolioLimits()
    stress_costs = stress_costs or {k: STRESS_COSTS[k] for k in ("spread_round_trip_pips", "commission_round_trip_pips", "slippage_round_trip_pips")}
    total_cost = sum(stress_costs.values())
    m8 = json.loads(mission8_path.read_text(encoding="utf-8"))
    m9 = json.loads(mission9_path.read_text(encoding="utf-8"))
    columns = ["strategy_id", "pair", "timeframe", "decision_timestamp", "entry_timestamp", "exit_timestamp",
               "direction", "entry_price", "stop_loss", "take_profit", "exit_price", "exit_reason",
               "gross_pnl_pips", "stop_distance_pips", "risk_reward", "position_units", "risk_amount",
               "regime", "session_utc", "signal_evidence"]
    frame = pd.read_csv(trade_path, usecols=columns, low_memory=False)
    for col in ("decision_timestamp", "entry_timestamp", "exit_timestamp"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="raise")
    for col in ("entry_price", "stop_loss", "take_profit", "exit_price", "gross_pnl_pips", "stop_distance_pips", "risk_reward", "position_units", "risk_amount"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    bounds = _research_split_bounds(m8) if m8.get("data_exclusions_and_provenance") else None
    frame = _assign_splits(frame, m8["results"], bounds)
    frame = frame[frame.split.isin(("train", "validation", "test"))].copy()
    strategies_found = sorted(frame.strategy_id.unique())
    pairs_found = sorted(frame.pair.unique())
    timeframes_found = sorted(frame.timeframe.unique())
    correlations, correlation_matrix = _correlation_map(m9)
    allocation = build_allocations(m9, correlations)
    comparison, accepted_logs, rejected_logs, curves, attribution_rows, signal_conflict_samples = [], [], [], [], [], []
    coordination_summary = {}
    for split in ("train", "validation", "test"):
        part = frame[frame.split == split].copy()
        candidates, conflict_rows, coord_counts = coordinate_signals(part, correlations)
        signal_conflict_samples.extend({"split": split, **row} for row in conflict_rows)
        coordination_summary[split] = {**coord_counts, "opposing_conflict_groups": len(conflict_rows),
                                       "source_independent_trade_rows": len(part)}
        comparison.extend(_independent_baselines(part, {"spread": 0, "commission": 0, "slippage": 0},
                                                 split, "gross_zero_cost", limits))
        comparison.extend(_independent_baselines(part, stress_costs, split,
                                                 "mission9_stress_assumption", limits))
        for scenario, cost_value in (("gross_zero_cost", 0.0), ("mission9_stress_assumption", total_cost)):
            for portfolio in PORTFOLIOS:
                result = simulate_portfolio(candidates, conflict_rows, portfolio=portfolio,
                                            allocation=allocation, correlations=correlations,
                                            limits=limits, costs_pips=cost_value, split=split,
                                            collect_rejections=(scenario == "mission9_stress_assumption"))
                tag = {"portfolio": portfolio, "cost_scenario": scenario, "split": split}
                comparison.append({**tag, **result["metrics"],
                                  "rejected_opportunity_count": sum(result["rejected_reason_counts"].values()),
                                  "rejected_reason_counts": result["rejected_reason_counts"]})
                accepted_logs.extend([{**tag, **trade} for trade in result["trades"]])
                rejected_logs.extend(result["rejected_sample"])
                curves.extend([{**tag, **point} for point in result["equity_curve"]])
                for dimension in ("strategy_id", "pair", "timeframe", "regime"):
                    for row in _attribution(result["trades"], dimension):
                        attribution_rows.append({**tag, "dimension": dimension, **row})
                attribution_rows.extend({**row, "dimension": "currency_exposure"} for row in
                                        _currency_exposure(result["trades"], portfolio=portfolio,
                                                           cost_scenario=scenario, split=split))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = {"json": output_dir / f"mission11_portfolio_{stamp}.json",
             "markdown": output_dir / f"mission11_portfolio_{stamp}.md",
             "comparison_csv": output_dir / f"mission11_comparison_{stamp}.csv",
             "attribution_csv": output_dir / f"mission11_attribution_{stamp}.csv",
             "correlation_csv": output_dir / f"mission11_validation_strategy_correlations_{stamp}.csv",
             "equity_csv": output_dir / f"mission11_equity_drawdown_{stamp}.csv",
             "currency_exposure_csv": output_dir / f"mission11_currency_exposure_{stamp}.csv",
             "signal_conflicts_csv": output_dir / f"mission11_signal_conflicts_{stamp}.csv",
             "trades_csv": output_dir / f"mission11_portfolio_trades_{stamp}.csv",
             "rejections_csv": output_dir / f"mission11_rejected_opportunities_sample_{stamp}.csv",
             "allocations_csv": output_dir / f"mission11_risk_allocations_{stamp}.csv"}
    risk_allocations = []
    for portfolio in PORTFOLIOS:
        for strategy in STRATEGIES:
            for market_regime, factor in (("all", allocation[portfolio][strategy] if portfolio != "regime_specialized" else None),):
                if portfolio == "regime_specialized":
                    for rg, val in allocation[portfolio][strategy].items():
                        risk_allocations.append({"portfolio": portfolio, "strategy_id": strategy, "regime": rg,
                                                 "risk_multiplier": val, "source_split": "validation"})
                else:
                    risk_allocations.append({"portfolio": portfolio, "strategy_id": strategy, "regime": market_regime,
                                             "risk_multiplier": factor, "source_split": "validation" if portfolio != "equal_risk" else "fixed equal risk"})
    diversification = _diversification_summary(comparison)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_files": {"mission8_json": str(mission8_path), "mission9_json": str(mission9_path), "mission8_trade_log": str(trade_path)},
              "source_fingerprints": {"mission8_json_sha256": _sha256(mission8_path),
                                      "mission9_json_sha256": _sha256(mission9_path),
                                      "trade_log_sha256": _sha256(trade_path),
                                      "mission8_data_fingerprints": m8.get("data_fingerprints", [])},
              "strategies_integrated": strategies_found, "strategy_count": len(strategies_found),
              "pairs_evaluated": pairs_found, "timeframes_evaluated": timeframes_found,
              "pair_timeframe_combinations": sorted({(x.pair, x.timeframe) for x in frame.itertuples()}),
              "source_trade_rows": int(sum(x["source_independent_trade_rows"] for x in coordination_summary.values())),
              "chronological_split_policy": "Mission 8 saved parquet boundaries; entry and exit must both remain within split; boundary-crossing/outside rows excluded.",
              "locked_test_policy": "Allocation maps use Mission 9 validation metrics/correlations only. Test metrics are descriptive holdout results and never feed allocation.",
              "allocation_evidence": allocation["validation_evidence"],
              "allocation_methods": {k: allocation[k] for k in PORTFOLIOS},
              "portfolio_limits": asdict(limits), "cost_scenarios": {"gross_zero_cost": {"total_round_trip_pips": 0, "status": "gross baseline"},
                                     "mission9_stress_assumption": {**stress_costs, "total_round_trip_pips": total_cost,
                                     "status": "assumption-based; commission/spread/slippage not broker-calibrated"}},
              "portfolio_configurations_tested": list(PORTFOLIOS),
              "independent_strategy_baseline_count": len(strategies_found),
              "coordination_counts_by_split": coordination_summary,
              "validation_strategy_correlation_matrix": correlation_matrix,
              "portfolio_results": comparison, "attribution": attribution_rows,
              "diversification_effect_descriptive": diversification,
              "accepted_trade_count": len(accepted_logs), "accepted_trade_log_csv": str(paths["trades_csv"]),
              "rejected_opportunity_sample_csv": str(paths["rejections_csv"]),
              "signal_conflict_sample_csv": str(paths["signal_conflicts_csv"]),
              "signal_conflict_sample_count": len(signal_conflict_samples),
              "equity_and_drawdown_curves_csv": str(paths["equity_csv"]),
              "rejected_opportunity_sample_limit": "At most 5,000 sampled risk rejections per portfolio/split/cost run; exact aggregate rejection counts are in comparison rows.",
              "risk_allocations": risk_allocations,
              "limitations": ["Mission 8 saved independently simulated opportunities are replayed; strategy signals/exits are not regenerated.",
                              "Mission 8 trade outcomes use next-bar entries and stop-first intrabar collision handling; portfolio uses those pre-resolved exits.",
                              "Currency exposures are tracked in allocated account-currency risk units by actual base/quote currency composition, not broker-converted notional lots.",
                              "Costs use an explicit Mission 9 stress assumption; current Mission 10 demo snapshots are insufficient for historical session-specific costs, and broker commission is unknown.",
                              "Portfolio drawdown is realized-equity based; no intrabar mark-to-market path is available in saved trade logs.",
                              "Drawdown and daily-loss guards reserve modeled stop risk plus costs for open trades, but gaps and end-of-data exits can still exceed configured thresholds.",
                              "Daily-return correlations from Mission 9 are sparse exit-day correlations and are used only as coarse validation-based allocation penalties.",
                              "No market-regime hypothesis is hardcoded as fact; all strategies remain available under all regimes, with bounded validation-derived risk scaling."],
              "safety": {"simulation_only": True, "broker_access": False, "orders_submitted": False,
                         "market_data_modified": False, "historical_downloads": False,
                         "model_training_or_promotion": False, "test_used_for_allocation": False},
              "report_files": {key: str(value) for key, value in paths.items()}}
    paths["json"].write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    pd.DataFrame(comparison).to_csv(paths["comparison_csv"], index=False)
    pd.DataFrame(attribution_rows).to_csv(paths["attribution_csv"], index=False)
    pd.DataFrame(correlation_matrix).to_csv(paths["correlation_csv"], index=False)
    pd.DataFrame(curves).to_csv(paths["equity_csv"], index=False)
    pd.DataFrame([r for r in attribution_rows if r.get("dimension") == "currency_exposure"]).to_csv(paths["currency_exposure_csv"], index=False)
    pd.DataFrame(signal_conflict_samples).to_csv(paths["signal_conflicts_csv"], index=False)
    pd.DataFrame(accepted_logs).to_csv(paths["trades_csv"], index=False)
    pd.DataFrame(rejected_logs).to_csv(paths["rejections_csv"], index=False)
    pd.DataFrame(risk_allocations).to_csv(paths["allocations_csv"], index=False)
    _write_markdown(report, paths["markdown"])
    return report


def refresh_independent_baselines(report_path: Path) -> dict[str, Any]:
    """Recompute the per-strategy baselines under the same shared limits.

    Kept separate so a report can correct/review baseline methodology without
    rerunning or changing any of the five combined portfolio simulations.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "signal_conflicts_csv" not in report["report_files"]:
        stamp = Path(report["report_files"]["markdown"]).stem.removeprefix("mission11_portfolio_")
        report["report_files"]["signal_conflicts_csv"] = str(Path(report["report_files"]["markdown"]).with_name(f"mission11_signal_conflicts_{stamp}.csv"))
    sources = report["source_files"]
    m8_path, m9_path, trade_path = (Path(sources[k]) for k in
                                   ("mission8_json", "mission9_json", "mission8_trade_log"))
    m8, m9 = json.loads(m8_path.read_text(encoding="utf-8")), json.loads(m9_path.read_text(encoding="utf-8"))
    columns = ["strategy_id", "pair", "timeframe", "decision_timestamp", "entry_timestamp", "exit_timestamp",
               "direction", "entry_price", "stop_loss", "take_profit", "exit_price", "exit_reason",
               "gross_pnl_pips", "stop_distance_pips", "risk_reward", "position_units", "risk_amount",
               "regime", "session_utc", "signal_evidence"]
    frame = pd.read_csv(trade_path, usecols=columns, low_memory=False)
    for col in ("decision_timestamp", "entry_timestamp", "exit_timestamp"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="raise")
    for col in ("gross_pnl_pips", "stop_distance_pips", "position_units", "risk_amount"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    bounds = _research_split_bounds(m8) if m8.get("data_exclusions_and_provenance") else None
    frame = _assign_splits(frame, m8["results"], bounds)
    correlations, _ = _correlation_map(m9)
    limits = PortfolioLimits(**report["portfolio_limits"])
    costs = report["cost_scenarios"]["mission9_stress_assumption"]
    costs = {k: float(costs[k]) for k in ("spread_round_trip_pips", "commission_round_trip_pips", "slippage_round_trip_pips")}
    rows = [r for r in report["portfolio_results"] if r["portfolio"] != "independent_strategy"]
    accepted_path = Path(report["report_files"]["trades_csv"])
    accepted = pd.read_csv(accepted_path, usecols=["portfolio", "cost_scenario", "split", "exit_timestamp", "net_r", "net_pnl_cash"])
    accepted["net_r"] = pd.to_numeric(accepted.net_r, errors="coerce")
    for (portfolio, scenario, split), group in accepted.groupby(["portfolio", "cost_scenario", "split"], sort=False):
        ordered = group.sort_values("exit_timestamp", kind="stable").net_r.to_numpy(float)
        cumulative = np.cumsum(ordered)
        dd_r = float((np.maximum.accumulate(np.r_[0.0, cumulative])[:-1] - cumulative).max()) if len(cumulative) else 0.0
        match = next((r for r in rows if r.get("portfolio") == portfolio and
                      r.get("cost_scenario") == scenario and r.get("split") == split), None)
        if match is not None:
            match["max_drawdown_r"] = dd_r
            pnl = pd.to_numeric(group.net_pnl_cash, errors="coerce").to_numpy(float)
            gains = float(pnl[pnl > 0].sum())
            losses = float(-pnl[pnl < 0].sum())
            match["profit_factor_net"] = gains / losses if losses else ("infinite" if gains else None)
    accepted_details = pd.read_csv(accepted_path,
                                   usecols=["portfolio", "cost_scenario", "split", "entry_timestamp",
                                            "exit_timestamp", "risk_cash", "currency_components"])
    exposure_rows = []
    for (portfolio, scenario, split), group in accepted_details.groupby(["portfolio", "cost_scenario", "split"], sort=False):
        values = []
        for row in group.to_dict("records"):
            row["currency_components"] = ast.literal_eval(row["currency_components"])
            values.append(row)
        exposure_rows.extend(_currency_exposure(values, portfolio=portfolio, cost_scenario=scenario, split=split))
    curve_path = Path(report["report_files"]["equity_csv"])
    curve_frame = pd.read_csv(curve_path, usecols=["portfolio", "cost_scenario", "split", "equity_net", "open_risk_cash"])
    for (portfolio, scenario, split), group in curve_frame.groupby(["portfolio", "cost_scenario", "split"], sort=False):
        utilization = group.open_risk_cash / np.maximum(group.equity_net * limits.max_total_open_risk, 1e-12)
        match = next((r for r in rows if r.get("portfolio") == portfolio and
                      r.get("cost_scenario") == scenario and r.get("split") == split), None)
        if match is not None:
            match["mean_open_risk_utilization_of_total_limit"] = float(utilization.mean())
            match["max_open_risk_utilization_of_total_limit"] = float(utilization.max())
    report["attribution"] = [r for r in report.get("attribution", []) if r.get("dimension") not in ("currency", "currency_exposure")]
    report["attribution"].extend({**row, "dimension": "currency_exposure"} for row in exposure_rows)
    for split in ("train", "validation", "test"):
        part = frame[frame.split == split]
        _, conflict_rows, _ = coordinate_signals(part, correlations)
        for conflict in conflict_rows:
            conflict["split"] = split
        conflict_path = Path(report["report_files"]["signal_conflicts_csv"])
        if split == "train":
            signal_conflicts = conflict_rows
        else:
            signal_conflicts.extend(conflict_rows)
        rows.extend(_independent_baselines(part, {"spread": 0, "commission": 0, "slippage": 0},
                                           split, "gross_zero_cost", limits))
        rows.extend(_independent_baselines(part, costs, split, "mission9_stress_assumption", limits))
    report["portfolio_results"] = rows
    report["diversification_effect_descriptive"] = _diversification_summary(rows)
    report["independent_baseline_count"] = sum(r.get("portfolio") == "independent_strategy" for r in rows)
    report["independent_baseline_protocol"] = "Per-strategy trade streams summarized in normalized R without shared portfolio constraints; portfolio versus baseline drawdown is compared in R, not fictional fixed-capital currency returns."
    report["baselines_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    pd.DataFrame(rows).to_csv(Path(report["report_files"]["comparison_csv"]), index=False)
    pd.DataFrame(report["attribution"]).to_csv(Path(report["report_files"]["attribution_csv"]), index=False)
    pd.DataFrame(exposure_rows).to_csv(Path(report["report_files"]["currency_exposure_csv"]), index=False)
    pd.DataFrame(signal_conflicts).to_csv(conflict_path, index=False)
    report["signal_conflict_sample_count"] = len(signal_conflicts)
    report["signal_conflict_sample_csv"] = str(conflict_path)
    report["report_files"]["signal_conflicts_csv"] = str(conflict_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    _write_markdown(report, Path(report["report_files"]["markdown"]))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = [r for r in report["portfolio_results"] if r.get("portfolio") in PORTFOLIOS]
    lines = ["# Mission 11 Multi-Strategy Portfolio Simulation", "",
             f"Generated UTC: {report['generated_at_utc']}", "",
             f"Integrated {report['strategy_count']} strategies across {len(report['pairs_evaluated'])} pairs and {len(report['timeframes_evaluated'])} timeframes.",
             "All strategies remain available in every regime; allocation methods adjust risk without selecting a single winner.", "",
             "## Portfolio comparison", "",
             "Returns below are simulated on the saved Mission 8 trade stream. Costed results use the explicitly assumed Mission 9 stress scenario.", "",
             "| Portfolio | Cost | Split | Trades | Net return | Expectancy R | PF | Max DD | Sharpe | Rejected |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def fmt(value, suffix=""):
            if value is None:
                return "n/a"
            if isinstance(value, (int, float)) and suffix == "%":
                return f"{value * 100:.2f}%"
            return f"{value:.4f}{suffix}" if isinstance(value, (int, float)) else str(value)
        lines.append(f"| {row['portfolio']} | {row['cost_scenario']} | {row['split']} | {row['trades']} | {fmt(row['net_return_fraction'], '%')} | {fmt(row['expectancy_r_net'])} | {fmt(row['profit_factor_net'])} | {fmt(row['max_drawdown_fraction'], '%')} | {fmt(row['sharpe_daily_annualized_365'])} | {row['rejected_opportunity_count']} |")
    lines.extend(["", "## Individual strategy baselines", "",
                  "Each saved strategy trade stream is summarized independently in normalized R, without applying shared portfolio limits. Portfolio drawdown is also expressed in R for comparison; currency return figures are not used to create unbounded fixed-capital baselines.", "",
                  "| Cost | Split | Median individual trades | Median individual max DD (R) | Equal-risk portfolio trades | Equal-risk portfolio max DD (R) |",
                  "|---|---|---:|---:|---:|---:|"])
    for item in report.get("diversification_effect_descriptive", []):
        lines.append(f"| {item['cost_scenario']} | {item['split']} | {item['median_individual_strategy_trades']:.1f} | {item['median_individual_strategy_max_drawdown_r']:.2f} | {item['equal_risk_portfolio_trades']} | {item['equal_risk_portfolio_max_drawdown_r']:.2f} |")
    lines.extend(["", "Comparisons are descriptive; currency composition and opportunity concurrency differ between individual and combined runs.", "",
                  "## Split and safety policy", "", report["chronological_split_policy"], "", report["locked_test_policy"], "",
                  "Validation-derived allocation weights were fixed before test evaluation. Test figures are descriptive and were not used to choose a portfolio.",
                  "Currency exposure is measured in allocated risk units applied to the pair's base and quote currencies; it is not a notional lot or broker margin estimate.", "",
                  "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", f"The conflict-sample CSV contains {report.get('signal_conflict_sample_count', 0)} simultaneous opposing-signal groups with strategy, direction, and evidence details.",
                  "Risk-limit rejection counts are exact by portfolio/split/scenario; the rejection CSV contains capped examples.", "",
                  "No profitability claim is made. Full attribution, curves, logs, allocation maps, rejection samples, fingerprints, and assumptions are in the JSON/CSV files."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    reports = ROOT / "reports/strategy-research"
    parser.add_argument("--mission8-json", type=Path, default=None)
    parser.add_argument("--mission9-json", type=Path, default=None)
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/strategy-research/mission11")
    parser.add_argument("--initial-balance", type=float, default=10_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.005)
    parser.add_argument("--max-open-risk", type=float, default=0.02)
    parser.add_argument("--max-pair-risk", type=float, default=0.0075)
    parser.add_argument("--max-currency-risk", type=float, default=0.01)
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--max-daily-loss", type=float, default=0.03)
    parser.add_argument("--max-drawdown", type=float, default=0.10)
    parser.add_argument("--stress-spread-pips", type=float, default=2.0)
    parser.add_argument("--stress-commission-pips", type=float, default=1.0)
    parser.add_argument("--stress-slippage-pips", type=float, default=0.5)
    args = parser.parse_args(argv)
    m8 = args.mission8_json or _latest(reports, "strategy_research_*.json")
    m9 = args.mission9_json or _latest(reports / "mission9", "mission9_cost_research_*.json")
    trades = args.trade_log or _latest(reports, "trade_log_*.csv")
    limits = PortfolioLimits(args.initial_balance, args.risk_per_trade, args.max_open_risk,
                             args.max_pair_risk, args.max_currency_risk, args.max_positions,
                             args.max_daily_loss, args.max_drawdown)
    costs = {"spread_round_trip_pips": args.stress_spread_pips,
             "commission_round_trip_pips": args.stress_commission_pips,
             "slippage_round_trip_pips": args.stress_slippage_pips}
    report = run_mission11(m8, m9, trades, args.output_dir, limits=limits, stress_costs=costs)
    print(f"Strategies={report['strategy_count']} pairs={len(report['pairs_evaluated'])} timeframes={report['timeframes_evaluated']}")
    for key, value in report["report_files"].items():
        print(f"{key}: {value}")
    print("Simulation only; no orders, data changes, training, or promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
