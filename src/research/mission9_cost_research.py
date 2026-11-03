"""Mission 9 cost sensitivity analysis over saved Mission 8 artifacts.

This module never regenerates strategy signals or writes to market data. Costs
are applied to the persisted gross trade outcomes as clearly labelled scenarios.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.trader.config import AppConfig
from src.trader.mt5_adapter import MT5DataAdapter
from src.trader.market_sessions import classify_utc_sessions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")
DEFAULT_SPREAD_SCENARIOS = (0.0, 1.0, 2.0)  # round-trip pips; assumptions, not broker measurements
DEFAULT_COMMISSION_SCENARIOS = (0.0, 0.5, 1.0)  # round-trip pips; assumptions
DEFAULT_SLIPPAGE_SCENARIOS = (0.0, 0.2, 0.5)  # round-trip pips; assumptions


@dataclass(frozen=True)
class CostScenario:
    spread_round_trip_pips: float
    commission_round_trip_pips: float
    slippage_round_trip_pips: float

    @property
    def name(self) -> str:
        return (f"spread_{self.spread_round_trip_pips:g}_commission_"
                f"{self.commission_round_trip_pips:g}_slippage_"
                f"{self.slippage_round_trip_pips:g}_pips_rt")


def _metrics(frame: pd.DataFrame, pnl_col: str) -> dict[str, Any]:
    pnl = pd.to_numeric(frame[pnl_col], errors="coerce").dropna().to_numpy(float)
    if not len(pnl):
        return {"trades": 0, "win_rate": None, "expectancy_r": None,
                "profit_factor": None, "max_drawdown_r": None,
                "max_consecutive_losses": 0, "total_r": 0.0, "gross_pips": 0.0}
    risk = pd.to_numeric(frame.loc[pd.to_numeric(frame[pnl_col], errors="coerce").notna(),
                                   "stop_distance_pips"], errors="coerce").to_numpy(float)
    risk = np.where(np.isfinite(risk) & (risk > 0), risk, np.nan)
    returns_r = pnl / risk
    valid = np.isfinite(returns_r)
    pnl, returns_r = pnl[valid], returns_r[valid]
    if not len(pnl):
        return {"trades": 0, "win_rate": None, "expectancy_r": None,
                "profit_factor": None, "max_drawdown_r": None,
                "max_consecutive_losses": 0, "total_r": 0.0, "gross_pips": 0.0}
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    equity = np.r_[0.0, np.cumsum(returns_r)]
    drawdown = np.maximum.accumulate(equity) - equity
    streak = best = 0
    for value in pnl:
        streak = streak + 1 if value < 0 else 0
        best = max(best, streak)
    pf = float(gains / losses) if losses else ("infinite" if gains else None)
    return {"trades": int(len(pnl)), "win_rate": float((pnl > 0).mean()),
            "expectancy_r": float(returns_r.mean()),
            "profit_factor": pf,
            "max_drawdown_r": float(drawdown.max()),
            "max_consecutive_losses": int(best), "total_r": float(returns_r.sum()),
            "gross_pips": float(pd.to_numeric(frame.loc[valid, "gross_pnl_pips"], errors="coerce").sum())}


def _research_split_bounds(payload: dict[str, Any]) -> dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    bounds = {}
    for item in payload.get("data_exclusions_and_provenance", []):
        source = Path(item.get("derived_file") or item["source_file"])
        frame = pd.read_parquet(source, columns=["time"])
        times = pd.to_datetime(frame.time, utc=True)
        n = len(times)
        i1, i2 = int(n * 0.60), int(n * 0.80)
        bounds[(source.stem.split("_")[0], source.stem.split("_")[-1])] = (
            times.iloc[0], times.iloc[i1], times.iloc[i2],
            times.iloc[-1] + pd.Timedelta(microseconds=1))
    return bounds


def _assign_splits(trades: pd.DataFrame, results: list[dict[str, Any]],
                   split_bounds: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] | None = None) -> pd.DataFrame:
    """Assign the saved, chronologically ordered trades from persisted split counts."""
    result_map = {(r["strategy_id"], r["pair"], r["timeframe"]): r for r in results}
    out: list[pd.DataFrame] = []
    keys = ["strategy_id", "pair", "timeframe"]
    for key, group in trades.groupby(keys, sort=False, dropna=False):
        record = result_map.get(tuple(key))
        if record is None:
            raise ValueError(f"No Mission 8 result metadata for trade group {key}")
        group = group.sort_values(["entry_timestamp", "exit_timestamp"], kind="stable").copy()
        if split_bounds and (key[1], key[2]) in split_bounds:
            start, validation_start, test_start, end = split_bounds[(key[1], key[2])]
            entry, exit_ = group.entry_timestamp, group.exit_timestamp
            labels = np.full(len(group), "purged_boundary_crossing_or_outside", dtype=object)
            labels[((entry >= start) & (exit_ < validation_start)).to_numpy()] = "train"
            labels[((entry >= validation_start) & (exit_ < test_start)).to_numpy()] = "validation"
            labels[((entry >= test_start) & (exit_ < end)).to_numpy()] = "test"
            group["split"] = labels
            out.append(group)
            continue
        counts = record.get("splits", {})
        expected = [int(counts.get(split, {}).get("trades", 0)) for split in ("train", "validation", "test")]
        if sum(expected) != len(group):
            raise ValueError(f"Saved split counts {expected} do not match {len(group)} trades for {key}")
        group["split"] = np.repeat(["train", "validation", "test"], expected)
        out.append(group)
    return pd.concat(out, ignore_index=True) if out else trades.assign(split=pd.Series(dtype=str))


def analyze_saved_results(json_path: Path, trade_path: Path,
                          scenarios: Iterable[CostScenario]) -> dict[str, Any]:
    """Calculate gross and scenario net metrics without running strategy code."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    trades = pd.read_csv(trade_path, low_memory=False)
    raw_trade_log_rows = len(trades)
    required = {"strategy_id", "pair", "timeframe", "entry_timestamp", "exit_timestamp",
                "direction", "gross_pnl_pips", "stop_distance_pips", "regime", "session_utc"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"Mission 8 trade log missing columns: {sorted(missing)}")
    trades["entry_timestamp"] = pd.to_datetime(trades.entry_timestamp, utc=True, errors="raise")
    trades["exit_timestamp"] = pd.to_datetime(trades.exit_timestamp, utc=True, errors="raise")
    trades["gross_pnl_pips"] = pd.to_numeric(trades.gross_pnl_pips, errors="raise")
    trades["stop_distance_pips"] = pd.to_numeric(trades.stop_distance_pips, errors="coerce")
    split_bounds = _research_split_bounds(payload) if payload.get("data_exclusions_and_provenance") else None
    trades = _assign_splits(trades, payload["results"], split_bounds)
    purged = trades[trades.split == "purged_boundary_crossing_or_outside"]
    purged_summary = (purged.groupby(["strategy_id", "pair", "timeframe"], sort=False).size()
                      .rename("purged_boundary_or_outside_trades").reset_index().to_dict("records"))
    trades = trades[trades.split != "purged_boundary_crossing_or_outside"].copy()
    rows: list[dict[str, Any]] = []
    scenario_list = list(scenarios)
    trade_groups = {(key[0], key[1], key[2], key[3]): group for key, group in trades.groupby(
        ["strategy_id", "pair", "timeframe", "split"], sort=False)}
    for result in payload["results"]:
        strategy, pair, timeframe = result["strategy_id"], result["pair"], result["timeframe"]
        for split in ("train", "validation", "test"):
            group = trade_groups.get((strategy, pair, timeframe, split), trades.iloc[0:0])
            gross = _metrics(group.assign(_pnl=group.gross_pnl_pips), "_pnl")
            rows.append({"strategy_id": strategy, "pair": pair, "timeframe": timeframe,
                         "split": split, "scenario": "gross_before_costs", "costs": None,
                         **gross})
            for scenario in scenario_list:
                cost_pips = (scenario.spread_round_trip_pips + scenario.commission_round_trip_pips
                             + scenario.slippage_round_trip_pips)
                tested = group.assign(_pnl=group.gross_pnl_pips - cost_pips)
                rows.append({"strategy_id": strategy, "pair": pair, "timeframe": timeframe,
                             "split": split, "scenario": scenario.name,
                             "costs": asdict(scenario), **_metrics(tested, "_pnl")})
    detail = pd.DataFrame(rows)
    combo_cols = ["strategy_id", "timeframe", "split", "scenario"]
    detail["profit_factor_numeric"] = pd.to_numeric(detail.profit_factor, errors="coerce")
    leaderboard = (detail.groupby(combo_cols, dropna=False)
                   .agg(pair_timeframe_combinations=("pair", "nunique"),
                        trades=("trades", "sum"),
                        median_expectancy_r=("expectancy_r", "median"),
                        positive_combinations=("expectancy_r", lambda s: int((s > 0).sum())),
                        median_profit_factor=("profit_factor_numeric", "median"),
                        median_max_drawdown_r=("max_drawdown_r", "median"),
                        mean_win_rate=("win_rate", "mean"),
                        worst_expectancy_r=("expectancy_r", "min"))
                   .reset_index())
    decision_scenario = max(scenario_list, key=lambda sc: sc.spread_round_trip_pips +
                            sc.commission_round_trip_pips + sc.slippage_round_trip_pips).name if scenario_list else ""
    candidate_assessment = []
    validation = detail[(detail.split == "validation") &
                        (detail.scenario == decision_scenario)]
    for strategy, group in validation.groupby("strategy_id", sort=False):
        positive_share = float((group.expectancy_r > 0).sum() / 21)
        median_r = float(group.expectancy_r.median()) if len(group) else None
        total_trades = int(group.trades.sum())
        median_combo_trades = float(group.trades.median()) if len(group) else 0.0
        if total_trades < 1000 or median_combo_trades < 25:
            label = "requires_more_validation_evidence"
        elif median_r is not None and median_r > 0 and positive_share >= 0.6 and total_trades >= 100:
            label = "validation_candidate_for_more_evidence"
        elif median_r is not None and median_r <= 0 and positive_share < 0.4:
            label = "weak_in_validation_under_sensitivity_scenario"
        else:
            label = "inconclusive_or_mixed"
        candidate_assessment.append({"strategy_id": strategy, "validation_scenario": decision_scenario,
                                     "median_pair_timeframe_expectancy_r": median_r,
                                     "positive_pair_timeframe_share": positive_share,
                                     "median_trades_per_pair_timeframe": median_combo_trades,
                                     "validation_trade_count": total_trades,
                                     "descriptive_status": label,
                                     "basis": "validation only; not an automatic strategy selection"})
    highest_cost = max(scenario_list, key=lambda sc: sc.spread_round_trip_pips +
                       sc.commission_round_trip_pips + sc.slippage_round_trip_pips) if scenario_list else None
    corr_scenarios = [CostScenario(0.0, 0.0, 0.0)]
    if highest_cost and highest_cost != corr_scenarios[0]:
        corr_scenarios.append(highest_cost)
    regimes = _grouped_breakdown(trades, corr_scenarios, "regime")
    sessions = _grouped_breakdown(trades, corr_scenarios, "session_utc")
    relationships = _relationship_summary(payload.get("signal_relationships", []))
    overlaps = _trade_overlap(trades)
    correlations = _return_correlations(trades, corr_scenarios)
    complementarity = _complementarity(payload.get("signal_relationships", []), overlaps,
                                       correlations, highest_cost.name if highest_cost else "")
    regime_frame = pd.DataFrame([x for x in regimes if x["split"] == "validation" and
                                 x["scenario"] == (highest_cost.name if highest_cost else "")])
    regime_specialization = []
    if not regime_frame.empty:
        for (strategy, regime), group in regime_frame.groupby(["strategy_id", "regime"], sort=False):
            median_r = float(group.expectancy_r.median()) if group.expectancy_r.notna().any() else None
            regime_specialization.append({"strategy_id": strategy, "regime": regime,
                                          "validation_pair_timeframe_count": int(len(group)),
                                          "median_expectancy_r": median_r,
                                          "positive_pair_timeframe_count": int((group.expectancy_r > 0).sum()),
                                          "descriptive_status": "regime_specific_signal_to_review" if median_r is not None and median_r > 0 and len(group) >= 3 else "not_enough_positive_validation_evidence"})
    return {"source_json": str(json_path), "source_trade_log": str(trade_path),
            "source_generated_at_utc": payload.get("generated_at_utc"),
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_trade_rows": int(raw_trade_log_rows),
            "evaluated_trade_rows": int(len(trades)),
            "strategies": len({x["strategy_id"] for x in payload.get("results", [])}),
            "pair_timeframe_combinations": int(trades[["pair", "timeframe"]].drop_duplicates().shape[0]),
            "cost_scenarios": [asdict(x) | {"name": x.name, "values_are_assumptions": True}
                               for x in scenario_list],
            "split_policy": "Mission 8 persisted split trade counts; chronological assignment; test period reported separately and excluded from candidate labels",
            "metrics": detail.to_dict(orient="records"),
            "purged_boundary_or_outside_trade_counts": purged_summary,
            "leaderboard": leaderboard.to_dict(orient="records"),
            "candidate_assessment_validation_only": candidate_assessment,
            "candidate_screen_policy": {
                "basis": "validation only under the highest-cost declared sensitivity scenario",
                "requires_more_evidence": "fewer than 1,000 total validation trades or median fewer than 25 trades per pair/timeframe",
                "validation_candidate_for_more_evidence": "median expectancy above zero, at least 60% of all 21 pair/timeframe combinations positive, and at least 1,000 validation trades",
                "weak_in_validation_under_sensitivity_scenario": "median expectancy at or below zero and fewer than 40% of all 21 pair/timeframe combinations positive",
                "other": "mixed or inconclusive; labels do not discard or promote strategies",
            },
            "market_regime": regimes, "session_utc": sessions,
            "regime_specialization_validation_only": regime_specialization,
            "signal_relationship_summary": relationships,
            "trade_overlap": overlaps, "daily_return_correlations": correlations,
            "complementarity_analysis_validation_only": complementarity,
            "complementarity_screen_policy": "Descriptive flag requires signal comparison in at least 3 pair/timeframe combinations, validation median absolute daily exit-return correlation below 0.3, and median trade-overlap share below 0.1; daily exit returns are sparse and not mark-to-market.",
            "limitations": ["Mission 8 gross P&L uses single-price OHLC fills; these cost deductions are fixed round-trip sensitivities, not replayed bid/ask fills.",
                            "Current MT5 spread snapshots are not historical trade-time spreads.",
                            "Connected MT5 tick timestamps were about two hours ahead of observation UTC, so no live spread sample passed timestamp validation and multi-session spread distributions remain unverified.",
                            "Daily return correlation uses realized trade exits with zero on days without an exit; it is not mark-to-market portfolio equity correlation.",
                            "Test metrics are descriptive holdout results and must not be used to tune or select parameters."]}


def _grouped_breakdown(trades: pd.DataFrame, scenarios: list[CostScenario], column: str) -> list[dict[str, Any]]:
    groups = ["strategy_id", "pair", "timeframe", "split", column]
    output = []
    for keys, frame in trades.groupby(groups, dropna=False, sort=False):
        base = dict(zip(groups, keys))
        for sc in scenarios:
            pnl = frame.gross_pnl_pips - sc.spread_round_trip_pips - sc.commission_round_trip_pips - sc.slippage_round_trip_pips
            output.append({**base, "scenario": sc.name, **_metrics(frame.assign(_pnl=pnl), "_pnl")})
    return output


def _relationship_summary(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (pair, tf), group in pd.DataFrame(items).groupby(["pair", "timeframe"], sort=False):
        coincident = pd.to_numeric(group.coincident_timestamps, errors="coerce").fillna(0)
        agrees = pd.to_numeric(group.directional_agreement_when_coincident, errors="coerce")
        output.append({"pair": pair, "timeframe": tf, "strategy_pairs": int(len(group)),
                       "coincident_signal_events_sum": int(coincident.sum()),
                       "agreement_fraction_weighted": float(np.average(agrees.fillna(0), weights=coincident)) if coincident.sum() else None,
                       "conflict_fraction_weighted": float(1 - np.average(agrees.fillna(0), weights=coincident)) if coincident.sum() else None,
                       "interpretation_counts": group.interpretation.value_counts().to_dict()})
    return output


def _trade_overlap(trades: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for (pair, tf, split), frame in trades.groupby(["pair", "timeframe", "split"], sort=False):
        strategies = sorted(frame.strategy_id.unique())
        for i, a in enumerate(strategies):
            aa = frame[frame.strategy_id == a][["entry_timestamp", "exit_timestamp"]].to_numpy()
            for b in strategies[i + 1:]:
                bb = frame[frame.strategy_id == b][["entry_timestamp", "exit_timestamp"]].to_numpy()
                b_starts = sorted(bb[:, 0]) if len(bb) else []
                b_ends = sorted(bb[:, 1]) if len(bb) else []
                overlaps = sum(bisect.bisect_right(b_starts, x[1]) -
                               bisect.bisect_left(b_ends, x[0]) for x in aa)
                output.append({"pair": pair, "timeframe": tf, "split": split,
                               "strategy_a": a, "strategy_b": b, "overlapping_trade_pairs": int(overlaps),
                               "trades_a": int(len(aa)), "trades_b": int(len(bb))})
    return output


def _return_correlations(trades: pd.DataFrame, scenarios: list[CostScenario]) -> list[dict[str, Any]]:
    output = []
    for (pair, timeframe, split), frame in trades.groupby(["pair", "timeframe", "split"], sort=False):
        for scenario in scenarios:
            work = frame.copy()
            cost = (scenario.spread_round_trip_pips + scenario.commission_round_trip_pips
                    + scenario.slippage_round_trip_pips)
            work["return_r"] = (work.gross_pnl_pips - cost) / work.stop_distance_pips
            work["date"] = work.exit_timestamp.dt.floor("D")
            daily = work.groupby(["date", "strategy_id"]).return_r.sum().unstack(fill_value=0)
            corr = daily.corr(min_periods=10)
            strategies = sorted(corr.columns)
            for i, a in enumerate(strategies):
                for b in strategies[i + 1:]:
                    value = corr.loc[a, b]
                    output.append({"pair": pair, "timeframe": timeframe, "split": split,
                                   "scenario": scenario.name, "strategy_a": a,
                                   "strategy_b": b,
                                   "daily_exit_return_correlation": float(value) if pd.notna(value) else None,
                                   "observed_exit_days": int(len(daily))})
    return output


def _complementarity(relationships: list[dict[str, Any]], overlaps: list[dict[str, Any]],
                     correlations: list[dict[str, Any]], scenario: str) -> list[dict[str, Any]]:
    signal_rows = []
    for row in relationships:
        a, b = sorted((row["strategy_a"], row["strategy_b"]))
        signal_rows.append({**row, "strategy_a": a, "strategy_b": b})
    signals = pd.DataFrame(signal_rows)
    if signals.empty:
        return []
    signals["coincident_timestamps"] = pd.to_numeric(signals.coincident_timestamps, errors="coerce").fillna(0)
    signals["agreement"] = pd.to_numeric(signals.directional_agreement_when_coincident, errors="coerce")
    keys = ["strategy_a", "strategy_b"]
    signal_summary = []
    for pair, frame in signals.groupby(keys, sort=False):
        n = frame.coincident_timestamps.sum()
        agree = float(np.average(frame.agreement.fillna(0), weights=frame.coincident_timestamps)) if n else None
        signal_summary.append({"strategy_a": pair[0], "strategy_b": pair[1],
                               "signal_combinations": int(len(frame)),
                               "coincident_signal_events": int(n),
                               "signal_agreement_fraction": agree,
                               "signal_conflict_fraction": 1 - agree if agree is not None else None})
    overlap_frame = pd.DataFrame([x for x in overlaps if x["split"] == "validation"])
    if not overlap_frame.empty:
        overlap_frame["overlap_share_of_smaller_trade_count"] = overlap_frame.apply(
            lambda x: x.overlapping_trade_pairs / min(x.trades_a, x.trades_b)
            if min(x.trades_a, x.trades_b) else 0.0, axis=1)
        overlap_summary = (overlap_frame.groupby(keys, sort=False)
                           .agg(validation_combinations=("pair", "size"),
                                median_trade_overlap_share=("overlap_share_of_smaller_trade_count", "median"))
                           .reset_index())
    else:
        overlap_summary = pd.DataFrame(columns=keys + ["validation_combinations", "median_trade_overlap_share"])
    corr_frame = pd.DataFrame([x for x in correlations if x["split"] == "validation" and x["scenario"] == scenario])
    if not corr_frame.empty:
        corr_summary = (corr_frame.groupby(keys, sort=False)
                        .agg(median_daily_return_correlation=("daily_exit_return_correlation", "median"),
                             correlation_combinations=("pair", "size"))
                        .reset_index())
    else:
        corr_summary = pd.DataFrame(columns=keys + ["median_daily_return_correlation", "correlation_combinations"])
    combined = pd.DataFrame(signal_summary).merge(overlap_summary, on=keys, how="left").merge(corr_summary, on=keys, how="left")
    labels = []
    for row in combined.itertuples(index=False):
        if (row.signal_combinations >= 3 and row.median_daily_return_correlation is not None
                and pd.notna(row.median_daily_return_correlation)
                and row.median_trade_overlap_share is not None and pd.notna(row.median_trade_overlap_share)
                and abs(row.median_daily_return_correlation) < 0.3
                and row.median_trade_overlap_share < 0.1):
            labels.append("potentially_complementary; validate further")
        elif row.signal_agreement_fraction is not None and row.signal_agreement_fraction >= 0.8:
            labels.append("high signal agreement; possible redundancy")
        else:
            labels.append("mixed or insufficient complementarity evidence")
    combined["descriptive_interpretation"] = labels
    combined["test_period_used"] = False
    records = combined.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                record[key] = None
    return records


def collect_spread_snapshot(adapter: MT5DataAdapter, pairs: Iterable[str] = DEFAULT_PAIRS,
                            *, samples: int = 1, interval_seconds: float = 5.0) -> dict[str, Any]:
    """Read bid/ask ticks. Invalid source timestamps are recorded, never concealed."""
    if samples < 1 or interval_seconds < 0:
        raise ValueError("samples must be positive and interval_seconds nonnegative")
    adapter.connect()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for sample_index in range(samples):
        for requested in pairs:
            observed = datetime.now(timezone.utc)
            try:
                broker_symbol = adapter.resolve_symbol(requested)
                tick = adapter.raw_tick(broker_symbol)
                info = adapter.symbol_info(broker_symbol)
                if tick is None or info is None:
                    raise RuntimeError("missing tick or symbol info")
                bid, ask = float(tick.bid), float(tick.ask)
                if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask >= bid):
                    raise RuntimeError("invalid bid/ask")
                normalized_time = adapter.normalize_tick_timestamp(tick, broker_symbol)
                freshness_seconds = (observed - normalized_time).total_seconds()
                validation_status = "future" if freshness_seconds < -5 else "stale" if freshness_seconds > adapter.max_quote_age_seconds else "valid"
                validation_error = None
                try:
                    adapter.quote_from_tick(broker_symbol, tick)
                except Exception as exc:
                    validation_error = str(exc)
                    if validation_status == "valid":
                        validation_status = "invalid_quote"
                pip = 0.01 if requested.endswith("JPY") else 0.0001
                session_tags = classify_utc_sessions(observed)
                rows.append({"requested_pair": requested, "broker_symbol": broker_symbol,
                             "sample_index": sample_index, "observed_at_utc": observed.isoformat(),
                             "observation_session_utc": session_tags[0] if session_tags else "OutsideConfiguredSessions",
                             "session_tags_utc": session_tags,
                             "raw_tick_time_seconds": int(getattr(tick, "time", 0) or 0),
                             "raw_tick_time_milliseconds": int(getattr(tick, "time_msc", 0) or 0),
                             "raw_tick_interpreted_as_utc": datetime.fromtimestamp(int(tick.time_msc) / 1000, timezone.utc).isoformat() if getattr(tick, "time_msc", 0) else None,
                             "normalized_tick_time_utc": normalized_time.isoformat(),
                             "tick_freshness_seconds": freshness_seconds,
                             "timestamp_status": validation_status,
                             "validation_error": validation_error,
                             "bid": bid, "ask": ask, "spread_price": ask - bid,
                             "spread_pips": (ask - bid) / pip,
                             "symbol_reported_spread_points": int(getattr(info, "spread", 0)),
                             "symbol_point": float(getattr(info, "point", 0)),
                             "symbol_digits": int(getattr(info, "digits", 0)),
                             "explicit_commission_field_names": [k for k in info._asdict() if "commission" in k.lower()] if hasattr(info, "_asdict") else []})
            except Exception as exc:
                errors.append({"pair": requested, "error": f"{type(exc).__name__}: {exc}"})
        if sample_index + 1 < samples and interval_seconds:
            time.sleep(interval_seconds)
    distributions = []
    df = pd.DataFrame(rows)
    if not df.empty:
        for (pair, session), group in df.groupby(["requested_pair", "observation_session_utc"], sort=False):
            good = group[group.timestamp_status == "valid"].spread_pips
            distributions.append({"pair": pair, "observation_session_utc": session,
                                  "samples": int(len(group)),
                                  "valid_timestamp_samples": int(len(good)),
                                  "min_spread_pips": float(good.min()) if len(good) else None,
                                  "median_spread_pips": float(good.median()) if len(good) else None,
                                  "p95_spread_pips": float(good.quantile(.95)) if len(good) else None,
                                  "max_spread_pips": float(good.max()) if len(good) else None,
                                  "raw_snapshot_median_spread_pips": float(group.spread_pips.median()),
                                  "all_raw_samples_spread_pips": group.spread_pips.tolist()})
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "server_timezone_config": adapter.server_timezone,
            "observation_window_seconds": max(0, samples - 1) * interval_seconds,
            "rows": rows, "distributions": distributions, "errors": errors,
            "commission": {"status": "not found in symbol metadata; terminal commission_blocked is account state, not a commission schedule"}}


def _write_reports(report: dict[str, Any], output_dir: Path, spread: dict[str, Any] | None) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"mission9_cost_research_{stamp}"
    json_path, csv_path, md_path = base.with_suffix(".json"), base.with_suffix(".csv"), base.with_suffix(".md")
    report["spread_observations"] = spread
    json_path.write_text(json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")
    pd.DataFrame(report["leaderboard"]).to_csv(csv_path, index=False)
    lines = ["# Mission 9 Strategy Validation and Transaction Cost Research", "",
             f"Generated: {report['analyzed_at_utc']}", "",
             f"Evaluated {report['evaluated_trade_rows']:,} in-period saved trades from {report['input_trade_rows']:,} trade-log rows across {report['strategies']} strategies and {report['pair_timeframe_combinations']} pair/timeframe combinations.", "",
             "## Cost inputs", "",
             "Mission 8 trades are gross single-price OHLC outcomes. The scenarios deduct separately specified round-trip spread, commission, and slippage assumptions. They are sensitivities, not historical executable fills.", "",
             "Commission schedule: unavailable in MT5 symbol metadata. Broker confirmation is required.", "",
             "## Holdout policy", "",
             "Train, validation, and test metrics remain separate. Candidate interpretation must use validation only; test is shown as a locked descriptive holdout and must not guide tuning.", "",
             "## Coverage and limitations", "",
             *[f"- {x}" for x in report["limitations"]], "",
             "Detailed per-combination, regime, session, overlap, and relationship data are in the JSON report.", "",
             "No strategy is automatically selected and these results do not establish profitability."]
    lines.extend(["", "## Validation assessment", "",
                  "The descriptive labels below use only the highest-cost declared sensitivity scenario within the validation split. They are screening labels, not deployment recommendations.",
             "Screen rules: more evidence is required below 1,000 validation trades or 25 median trades per pair/timeframe; a candidate screen requires positive median expectancy and positive expectancy in at least 60% of the full 21 combinations. These are descriptive thresholds only.",
                  "", "| Strategy | Validation label | Median expectancy (R) | Positive pair/timeframe share | Validation trades |",
                  "|---|---|---:|---:|---:|"])
    for row in report["candidate_assessment_validation_only"]:
        med = "n/a" if row["median_pair_timeframe_expectancy_r"] is None else f"{row['median_pair_timeframe_expectancy_r']:.4f}"
        lines.append(f"| {row['strategy_id']} | {row['descriptive_status']} | {med} | {row['positive_pair_timeframe_share']:.1%} | {row['validation_trade_count']} |")
    lines.extend(["", "No strategy meets the validation candidate threshold under the highest-cost sensitivity scenario." if not any(row["descriptive_status"] == "validation_candidate_for_more_evidence" for row in report["candidate_assessment_validation_only"]) else "At least one strategy meets the descriptive validation screen; independent evidence is still required."])
    regime_hits = [x for x in report.get("regime_specialization_validation_only", [])
                   if x["descriptive_status"] == "regime_specific_signal_to_review"]
    lines.extend(["", "## Regime-specific validation evidence", "",
                  "| Strategy | Regime | Pair/timeframe observations | Median expectancy (R) |",
                  "|---|---|---:|---:|"])
    for row in regime_hits:
        lines.append(f"| {row['strategy_id']} | {row['regime']} | {row['validation_pair_timeframe_count']} | {row['median_expectancy_r']:.4f} |")
    if not regime_hits:
        lines.append("No regime met the descriptive review threshold; regime-level metrics remain available in JSON.")
    if spread:
        lines.extend(["", "## Raw MT5 spread observations (cumulative)", "",
                      "| Pair | Session by observation time (UTC) | Samples | Raw spread pips | Valid tick-time samples |",
                      "|---|---|---:|---:|---:|"])
        for row in spread.get("cumulative_session_distributions", spread.get("distributions", [])):
            vals = (f"median {row['raw_snapshot_median_spread_pips']:.3f}; "
                    f"range {row.get('raw_min_spread_pips', min(row.get('all_raw_samples_spread_pips', [0]))):.3f}-"
                    f"{row.get('raw_max_spread_pips', max(row.get('all_raw_samples_spread_pips', [0]))):.3f}"
                    if "raw_snapshot_median_spread_pips" in row else
                    ", ".join(f"{v:.3f}" for v in row["all_raw_samples_spread_pips"]))
            lines.append(f"| {row['pair']} | {row['observation_session_utc']} | {row['samples']} | {vals} | {row['valid_timestamp_samples']} |")
        lines.append("Tick timestamp failures mean these values are recorded observations only; they do not establish valid session distributions or historical spreads. UTC session labels use local observation time; only sessions with recorded samples appear.")
    comps = [row for row in report.get("complementarity_analysis_validation_only", [])
             if row["descriptive_interpretation"] == "potentially_complementary; validate further"]
    lines.extend(["", "## Complementarity screen", "",
                  f"Potentially complementary pairs flagged for review: {len(comps)}. Signal disagreement, trade overlap, and daily exit-return correlations are available in the JSON; labels are descriptive and test-period independent."])
    if spread:
        lines.extend(["", "## MT5 spread snapshot", "",
                      f"Captured at {spread['captured_at_utc']}; window {spread['observation_window_seconds']} seconds.",
                      "Only samples with valid source timestamps contribute to session distribution summaries."])
        if spread["errors"]:
            lines.extend(["", "Errors:", *[f"- {e['pair']}: {e['error']}" for e in spread["errors"]]])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Mission 8 JSON; defaults to newest report")
    parser.add_argument("--trades", type=Path, help="Mission 8 trade log; defaults to newest report")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "strategy-research" / "mission9")
    parser.add_argument("--collect-spreads", action="store_true", help="Read current MT5 ticks; never sends orders")
    parser.add_argument("--spread-samples", type=int, default=1)
    parser.add_argument("--sample-interval-seconds", type=float, default=30.0)
    parser.add_argument("--spread-history", type=Path,
                        default=ROOT / "reports" / "strategy-research" / "mission9" / "spread_observations.jsonl",
                        help="Append quote observations so distributions can accumulate across session runs")
    args = parser.parse_args(argv)
    reports = ROOT / "reports" / "strategy-research"
    js = args.json or max(reports.glob("strategy_research_*.json"), key=lambda p: p.stat().st_mtime)
    trades = args.trades or max(reports.glob("trade_log_*.csv"), key=lambda p: p.stat().st_mtime)
    scenarios = [CostScenario(s, c, l) for s in DEFAULT_SPREAD_SCENARIOS
                 for c in DEFAULT_COMMISSION_SCENARIOS for l in DEFAULT_SLIPPAGE_SCENARIOS]
    print(f"Reading saved Mission 8 trades from {trades}", flush=True)
    report = analyze_saved_results(js, trades, scenarios)
    spread = None
    if args.collect_spreads:
        cfg = AppConfig.from_env()
        adapter = MT5DataAdapter(terminal_path=cfg.mt5_terminal_path,
                                 server_timezone=cfg.mt5_server_timezone,
                                 max_quote_age_seconds=cfg.max_quote_age_seconds)
        try:
            spread = collect_spread_snapshot(adapter, cfg.pairs, samples=args.spread_samples,
                                              interval_seconds=args.sample_interval_seconds)
        except Exception as exc:
            spread = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "rows": [],
                      "errors": [{"pair": "*", "error": f"{type(exc).__name__}: {exc}"}],
                      "commission": {"status": "unavailable"}}
        finally:
            adapter.close()
        if spread.get("rows"):
            args.spread_history.parent.mkdir(parents=True, exist_ok=True)
            with args.spread_history.open("a", encoding="utf-8") as history:
                for item in spread["rows"]:
                    history.write(json.dumps(item, allow_nan=False) + "\n")
            history_frame = pd.DataFrame(
                json.loads(line) for line in args.spread_history.read_text(encoding="utf-8").splitlines()
                if line.strip())
            cumulative = []
            for (pair, session), group in history_frame.groupby(
                    ["requested_pair", "observation_session_utc"], sort=False):
                valid = group[group.timestamp_status == "valid"].spread_pips
                cumulative.append({"pair": pair, "observation_session_utc": session,
                                   "samples": int(len(group)), "valid_timestamp_samples": int(len(valid)),
                                   "raw_snapshot_median_spread_pips": float(group.spread_pips.median()),
                                   "raw_min_spread_pips": float(group.spread_pips.min()),
                                   "raw_max_spread_pips": float(group.spread_pips.max()),
                                   "valid_timestamp_median_spread_pips": float(valid.median()) if len(valid) else None,
                                   "first_observed_at_utc": str(group.observed_at_utc.min()),
                                   "last_observed_at_utc": str(group.observed_at_utc.max())})
            spread["history_file"] = str(args.spread_history)
            spread["cumulative_session_distributions"] = cumulative
    written = _write_reports(report, args.output_dir, spread)
    print(f"Saved: {written[0]}\nSaved: {written[1]}\nSaved: {written[2]}")
    print(f"Evaluated {report['evaluated_trade_rows']} of {report['input_trade_rows']} trade-log rows; {report['strategies']} strategies; {report['pair_timeframe_combinations']} combinations.")
    if spread:
        print(f"Spread ticks: {len(spread.get('rows', []))}; errors: {len(spread.get('errors', []))}; commission schedule: unavailable from symbol metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
