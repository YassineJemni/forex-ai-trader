"""Pair/timeframe-scoped statistical evidence from saved Mission 8 trades.

This evaluator reads timestamps from the source candles to reconstruct the
existing chronological boundaries, but only parses trade outcomes before the
validation end. It never regenerates signals or evaluates post-validation P&L.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import mission9_cost_research as mission9
from . import multi_strategy_engine
from .experiment_store import PROJECT_ROOT
from .predictability import _holm_adjust


ROOT = PROJECT_ROOT
DEFAULT_MISSION8 = ROOT / "reports" / "strategy-research" / "strategy_research_20261103T092811Z.json"
DEFAULT_TRADES = ROOT / "reports" / "strategy-research" / "trade_log_20261103T092811Z.csv"
DEFAULT_MISSION9 = ROOT / "reports" / "strategy-research" / "mission9" / "mission9_cost_research_20261103T101819Z.json"
DEFAULT_MISSION11 = ROOT / "reports" / "strategy-research" / "mission11" / "mission11_portfolio_20261103T114226Z.json"
DEFAULT_OUTPUT = ROOT / "reports" / "strategy-research" / "mission13"

PAIRS = multi_strategy_engine.PAIRS
TIMEFRAMES = multi_strategy_engine.TIMEFRAMES
STRATEGY_IDS = tuple(x.strategy_id for x in multi_strategy_engine.strategy_library())
EXPECTED_FAMILY_SIZE = len(STRATEGY_IDS) * len(PAIRS) * len(TIMEFRAMES)
MIN_VALIDATION_TRADES = 25
MIN_SUBPERIOD_TRADES = 5
VALIDATION_SUBPERIODS = 3
BLOCK_LENGTH_TRADES = 5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
STRESS_COST_SCENARIO = "spread_2_commission_1_slippage_0.5_pips_rt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _cost_scenarios() -> list[mission9.CostScenario]:
    return [
        mission9.CostScenario(spread, commission, slippage)
        for spread in mission9.DEFAULT_SPREAD_SCENARIOS
        for commission in mission9.DEFAULT_COMMISSION_SCENARIOS
        for slippage in mission9.DEFAULT_SLIPPAGE_SCENARIOS
    ]


def _resolve_source(item: Mapping[str, Any], data_dir: Path | None) -> Path:
    candidate = Path(item.get("derived_file") or item.get("source_file", ""))
    if candidate.is_file():
        return candidate
    if data_dir is not None and candidate.name:
        fallback = data_dir / candidate.name
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"Mission 8 candle source is unavailable: {candidate}")


def load_split_boundaries(
    mission8: Mapping[str, Any],
    *,
    data_dir: Path | None = None,
) -> dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Reconstruct the original 60/20/20 split using timestamp columns only."""
    bounds = {}
    for item in mission8.get("data_exclusions_and_provenance", []):
        source = _resolve_source(item, data_dir)
        pair, timeframe = source.stem.rsplit("_", 1)
        if pair not in PAIRS or timeframe not in TIMEFRAMES:
            raise ValueError(f"unsupported Mission 8 source name: {source.name}")
        times = pd.to_datetime(pd.read_parquet(source, columns=["time"]).time, utc=True, errors="raise")
        if len(times) < 10 or times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError(f"invalid chronological timestamp source: {source}")
        i1, i2 = int(len(times) * 0.60), int(len(times) * 0.80)
        bounds[(pair, timeframe)] = (
            times.iloc[0], times.iloc[i1], times.iloc[i2],
            times.iloc[-1] + pd.Timedelta(microseconds=1),
        )
    expected_markets = {(pair, tf) for pair in PAIRS for tf in TIMEFRAMES}
    if set(bounds) != expected_markets:
        missing = sorted(expected_markets - set(bounds))
        raise ValueError(f"Mission 8 boundary metadata does not cover all markets: {missing}")
    return bounds


TRADE_COLUMNS = (
    "strategy_id", "pair", "timeframe", "decision_timestamp", "entry_timestamp",
    "exit_timestamp", "direction", "gross_pnl_pips", "stop_distance_pips",
    "regime", "session_utc",
)


def load_development_trades(
    trade_path: Path,
    boundaries: Mapping[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read train/validation trade outcomes only; skip post-validation P&L fields."""
    accepted: list[dict[str, Any]] = []
    counts = {"source_rows": 0, "training_rows": 0, "validation_rows": 0,
              "post_validation_rows_skipped": 0, "boundary_crossing_or_outside_rows": 0,
              "invalid_outcome_rows": 0}
    with trade_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty trade log: {trade_path}") from exc
        index = {name: header.index(name) for name in TRADE_COLUMNS if name in header}
        missing = set(TRADE_COLUMNS) - set(index)
        if missing:
            raise ValueError(f"Mission 8 trade log missing columns: {sorted(missing)}")
        for row in reader:
            counts["source_rows"] += 1
            strategy = row[index["strategy_id"]]
            pair, timeframe = row[index["pair"]], row[index["timeframe"]]
            if strategy not in STRATEGY_IDS:
                raise ValueError(f"trade log contains an unknown strategy ID: {strategy}")
            market_bounds = boundaries.get((pair, timeframe))
            if market_bounds is None:
                counts["boundary_crossing_or_outside_rows"] += 1
                continue
            start, validation_start, validation_end, _ = market_bounds
            entry = pd.Timestamp(row[index["entry_timestamp"]])
            exit_time = pd.Timestamp(row[index["exit_timestamp"]])
            if entry.tzinfo is None or exit_time.tzinfo is None:
                raise ValueError("trade timestamps must be timezone-aware")
            entry, exit_time = entry.tz_convert("UTC"), exit_time.tz_convert("UTC")
            if entry >= validation_end or exit_time >= validation_end:
                counts["post_validation_rows_skipped"] += 1
                continue
            if entry >= start and exit_time < validation_start:
                split = "train"
            elif entry >= validation_start and exit_time < validation_end:
                split = "validation"
            else:
                counts["boundary_crossing_or_outside_rows"] += 1
                continue
            try:
                gross = float(row[index["gross_pnl_pips"]])
                stop = float(row[index["stop_distance_pips"]])
                if not np.isfinite(gross) or not np.isfinite(stop) or stop <= 0:
                    raise ValueError("nonfinite outcome or nonpositive stop")
                if row[index["direction"]].upper() not in {"BUY", "SELL"}:
                    raise ValueError("trade direction must be BUY or SELL")
            except (TypeError, ValueError, OverflowError):
                counts["invalid_outcome_rows"] += 1
                continue
            record = {column: row[index[column]] for column in TRADE_COLUMNS}
            record.update({
                "entry_timestamp": entry, "exit_timestamp": exit_time,
                "decision_timestamp": pd.Timestamp(record["decision_timestamp"]).tz_convert("UTC"),
                "gross_pnl_pips": gross, "stop_distance_pips": stop, "split": split,
            })
            accepted.append(record)
            counts["training_rows" if split == "train" else "validation_rows"] += 1
    return pd.DataFrame(accepted, columns=[*TRADE_COLUMNS, "split"]), counts


def _bootstrap_mean(
    values: Sequence[float],
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Pointwise circular moving-block interval and centered one-sided test."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or not len(sample) or not np.isfinite(sample).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if block_length < 1 or block_length > len(sample) or n_resamples < 100:
        raise ValueError("invalid bootstrap block length or resample count")
    n_blocks = int(np.ceil(len(sample) / block_length))
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(sample), size=(n_resamples, n_blocks))
    indices = ((starts[:, :, None] + offsets[None, None, :]) % len(sample))
    indices = indices.reshape(n_resamples, -1)[:, :len(sample)]
    estimates = sample[indices].mean(axis=1)
    observed = float(sample.mean())
    centered_null_means = (sample - observed)[indices].mean(axis=1)
    p_value = (1 + int(np.count_nonzero(centered_null_means >= observed))) / (n_resamples + 1)
    lower, upper = np.quantile(estimates, [BOOTSTRAP_ALPHA / 2, 1 - BOOTSTRAP_ALPHA / 2])
    return {
        "mean": observed, "lower": float(lower), "upper": float(upper),
        "bootstrap_one_sided_p_value": float(p_value),
        "sample_trades": int(len(sample)), "block_length_trades": int(block_length),
        "n_resamples": int(n_resamples), "alpha": BOOTSTRAP_ALPHA, "seed": int(seed),
        "method": "circular_moving_block_percentile_ci_and_centered_null_test",
    }


def _seed_for(scope: tuple[str, str, str]) -> int:
    token = "|".join(("mission13.3", *scope, STRESS_COST_SCENARIO))
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")


def validate_saved_split_counts(results: Sequence[Mapping[str, Any]], trades: pd.DataFrame) -> None:
    """Verify source split counts without consulting post-validation outcomes."""
    actual = (trades.groupby(["strategy_id", "pair", "timeframe", "split"], sort=False)
              .size().to_dict())
    for row in results:
        counts = row.get("splits")
        if not isinstance(counts, Mapping):
            continue
        for split in ("train", "validation"):
            expected = int((counts.get(split) or {}).get("trades", 0))
            key = (row["strategy_id"], row["pair"], row["timeframe"], split)
            observed = int(actual.get(key, 0))
            if observed != expected:
                raise ValueError(
                    f"Mission 8 {split} trade count mismatch for {key}: "
                    f"source={expected}, reconstructed={observed}"
                )


def _trade_metrics(frame: pd.DataFrame, cost_pips: float, *, min_stratified_trades: int = 10) -> dict[str, Any]:
    pnl = frame.gross_pnl_pips.to_numpy(dtype=float) - cost_pips
    risk = frame.stop_distance_pips.to_numpy(dtype=float)
    r_values = pnl / risk
    metrics = mission9._metrics(frame.assign(_net_pnl=pnl), "_net_pnl")
    return {
        **metrics,
        "mean_outcome_pips": float(pnl.mean()) if len(pnl) else None,
        "median_outcome_pips": float(np.median(pnl)) if len(pnl) else None,
        "mean_outcome_r": float(r_values.mean()) if len(r_values) else None,
        "median_outcome_r": float(np.median(r_values)) if len(r_values) else None,
        "minimum_stratified_sample_met": bool(len(frame) >= min_stratified_trades),
    }


def _validation_periods(
    trades: pd.DataFrame,
    boundary: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
    cost_pips: float,
) -> list[dict[str, Any]]:
    _, validation_start, validation_end, _ = boundary
    span = validation_end - validation_start
    output = []
    for index in range(VALIDATION_SUBPERIODS):
        start = validation_start + span * index / VALIDATION_SUBPERIODS
        end = validation_start + span * (index + 1) / VALIDATION_SUBPERIODS
        period = trades.loc[trades.entry_timestamp.ge(start) & trades.entry_timestamp.lt(end)]
        metrics = _trade_metrics(period, cost_pips)
        output.append({
            "period_index": index + 1, "start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "trades": int(len(period)), "expectancy_r": metrics["expectancy_r"],
            "positive_expectancy": bool(metrics["expectancy_r"] is not None and metrics["expectancy_r"] > 0),
            "minimum_sample_met": len(period) >= MIN_SUBPERIOD_TRADES,
        })
    return output


def _stratified_metrics(
    trades: pd.DataFrame,
    scenario: mission9.CostScenario,
) -> list[dict[str, Any]]:
    cost = scenario.spread_round_trip_pips + scenario.commission_round_trip_pips + scenario.slippage_round_trip_pips
    rows = []
    for dimension in ("regime", "session_utc", "direction"):
        for key, frame in trades.groupby(["strategy_id", "pair", "timeframe", "split", dimension],
                                         sort=True, dropna=False):
            ids = key[:4]
            metrics = _trade_metrics(frame, cost)
            rows.append({
                "strategy_id": ids[0], "pair": ids[1], "timeframe": ids[2], "split": ids[3],
                "dimension": dimension, "category": str(key[4]), "scenario": scenario.name,
                **metrics,
            })
    return rows


def evaluate_exact_scope(
    results: Sequence[Mapping[str, Any]],
    trades: pd.DataFrame,
    boundaries: Mapping[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Evaluate every declared Mission 8 strategy/market cell on development data."""
    cells_by_key = {
        (str(row["strategy_id"]), str(row["pair"]), str(row["timeframe"])): dict(row)
        for row in results
    }
    if len(cells_by_key) != len(results):
        raise ValueError("duplicate exact-scope Mission 8 result rows")
    if set(STRATEGY_IDS) - {key[0] for key in cells_by_key}:
        raise ValueError("Mission 8 results do not preserve all 14 strategy engines")
    expected_cells = {(strategy, pair, timeframe)
                      for strategy in STRATEGY_IDS for pair in PAIRS for timeframe in TIMEFRAMES}
    if set(cells_by_key) != expected_cells:
        raise ValueError("Mission 8 results do not exactly cover all 14×7×3 cells")
    scenario_list = _cost_scenarios()
    stress = next((scenario for scenario in scenario_list if scenario.name == STRESS_COST_SCENARIO), None)
    if stress is None or len(scenario_list) != 27:
        raise ValueError("declared Mission 9 cost scenario set is incomplete")
    stress_cost = (stress.spread_round_trip_pips + stress.commission_round_trip_pips
                   + stress.slippage_round_trip_pips)
    metric_rows: list[dict[str, Any]] = []
    stratified_rows = _stratified_metrics(
        trades.loc[trades.split.isin(("train", "validation"))], stress)
    trade_frequency: dict[tuple[str, str, str, str], float | None] = {}
    validation_trade_groups = {
        (key[0], key[1], key[2]): group.sort_values("entry_timestamp", kind="stable")
        for key, group in trades.loc[trades.split == "validation"].groupby(
            ["strategy_id", "pair", "timeframe"], sort=False)
    }
    development_trade_groups = {
        key: group
        for key, group in trades.loc[trades.split.isin(("train", "validation"))].groupby(
            ["strategy_id", "pair", "timeframe", "split"], sort=False)
    }
    inference_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for scope in sorted(cells_by_key):
        strategy, pair, timeframe = scope
        for split in ("train", "validation"):
            group = development_trade_groups.get((*scope, split), trades.iloc[0:0])
            _, validation_start, validation_end, _ = boundaries[(pair, timeframe)]
            split_start, split_end = (
                (boundaries[(pair, timeframe)][0], validation_start)
                if split == "train" else (validation_start, validation_end)
            )
            duration_days = (split_end - split_start).total_seconds() / 86400
            trade_frequency[(*scope, split)] = (
                float(30 * len(group) / duration_days) if duration_days > 0 else None)
            for scenario in scenario_list:
                cost = scenario.spread_round_trip_pips + scenario.commission_round_trip_pips + scenario.slippage_round_trip_pips
                metric_rows.append({
                    "strategy_id": strategy, "pair": pair, "timeframe": timeframe,
                    "split": split, "scenario": scenario.name,
                    "spread_round_trip_pips_assumed": scenario.spread_round_trip_pips,
                    "commission_round_trip_pips_assumed": scenario.commission_round_trip_pips,
                    "slippage_round_trip_pips_assumed": scenario.slippage_round_trip_pips,
                    "assumptions_are_broker_calibrated": False,
                    "trades_per_30_calendar_days": trade_frequency[(*scope, split)],
                    "signal_frequency_status": "unavailable: Mission 8 signal counts are not split-scoped",
                    **_trade_metrics(group, cost),
                })
        validation = validation_trade_groups.get(scope, trades.iloc[0:0])
        periods = _validation_periods(validation, boundaries[(pair, timeframe)], stress_cost)
        metrics = _trade_metrics(validation, stress_cost)
        if len(validation) >= MIN_VALIDATION_TRADES:
            bootstrap = _bootstrap_mean(
                (validation.gross_pnl_pips.to_numpy(dtype=float) - stress_cost)
                / validation.stop_distance_pips.to_numpy(dtype=float),
                block_length=min(BLOCK_LENGTH_TRADES, len(validation)),
                n_resamples=n_resamples, seed=_seed_for(scope),
            )
        else:
            bootstrap = {
                "mean": metrics["expectancy_r"], "lower": None, "upper": None,
                "bootstrap_one_sided_p_value": 1.0, "sample_trades": len(validation),
                "block_length_trades": min(BLOCK_LENGTH_TRADES, len(validation)) if len(validation) else None,
                "n_resamples": n_resamples, "alpha": BOOTSTRAP_ALPHA, "seed": _seed_for(scope),
                "method": "not_run_insufficient_validation_sample",
            }
        inference = {
            "strategy_id": strategy, "pair": pair, "timeframe": timeframe,
            "validation_trades": int(len(validation)),
            "validation_expectancy_r": metrics["expectancy_r"],
            "validation_win_rate": metrics["win_rate"],
            "validation_profit_factor": metrics["profit_factor"],
            "validation_max_drawdown_r": metrics["max_drawdown_r"],
            "validation_mean_outcome_r": metrics["mean_outcome_r"],
            "validation_median_outcome_r": metrics["median_outcome_r"],
            "validation_positive_subperiod_share": float(sum(x["positive_expectancy"] for x in periods) / len(periods)),
            "validation_subperiods": periods,
            "train_trades": int(len(development_trade_groups.get((*scope, "train"), trades.iloc[0:0]))),
            "train_expectancy_r_at_stress_cost": _trade_metrics(
                development_trade_groups.get((*scope, "train"), trades.iloc[0:0]),
                stress_cost)["expectancy_r"],
            "bootstrap": bootstrap,
        }
        inference_rows.append(inference)
        cells.append({
            "strategy_id": strategy, "pair": pair, "timeframe": timeframe,
            "validation_metrics": metrics, "training_metrics_at_stress_cost": inference["train_expectancy_r_at_stress_cost"],
            "validation_periods": periods, "validation_bootstrap": bootstrap,
            "validation_trades_per_30_calendar_days": trade_frequency.get((*scope, "validation")),
            "validation_signal_frequency_status": "unavailable: Mission 8 signal counts are not split-scoped",
        })
    adjusted = _holm_adjust(np.asarray(
        [row["bootstrap"]["bootstrap_one_sided_p_value"] for row in inference_rows], dtype=float))
    classifications: dict[str, int] = {}
    for cell, infer, adjusted_p in zip(cells, inference_rows, adjusted):
        infer["holm_adjusted_p_value"] = float(adjusted_p)
        infer["holm_significant_at_0_05"] = bool(adjusted_p <= 0.05)
        n = infer["validation_trades"]
        periods = infer["validation_subperiods"]
        period_sample_ok = all(period["minimum_sample_met"] for period in periods)
        positive_periods = sum(period["positive_expectancy"] for period in periods)
        stable = (n >= MIN_VALIDATION_TRADES and period_sample_ok
                  and positive_periods >= 2 and (infer["validation_expectancy_r"] or 0) > 0)
        train_positive = (infer["train_expectancy_r_at_stress_cost"] is not None
                          and infer["train_expectancy_r_at_stress_cost"] > 0
                          and infer["train_trades"] >= MIN_VALIDATION_TRADES)
        if stable:
            stability_status = "stable_pair_timeframe_validation"
        else:
            stability_status = "unstable_or_insufficient_validation"
        if n < MIN_VALIDATION_TRADES or not period_sample_ok:
            classification = "INSUFFICIENT EVIDENCE"
        elif (infer["validation_expectancy_r"] or 0) <= 0:
            classification = "NEGATIVE EVIDENCE"
        elif stable and train_positive and adjusted_p <= 0.05:
            classification = "VALIDATION CANDIDATE"
        else:
            classification = "PROMISING — REQUIRES FURTHER VALIDATION"
        cell["validation_bootstrap"]["holm_adjusted_p_value"] = float(adjusted_p)
        cell["pair_timeframe_stability_status"] = stability_status
        cell["classification"] = classification
        cell["training_positive_at_stress_cost"] = bool(train_positive)
        infer["pair_timeframe_stability_status"] = stability_status
        infer["classification"] = classification
        classifications[classification] = classifications.get(classification, 0) + 1

    validation_trades = trades.loc[trades.split == "validation"].copy()
    if len(validation_trades):
        overlap_rows = mission9._trade_overlap(validation_trades)
        correlations = mission9._return_correlations(validation_trades, [stress])
    else:
        overlap_rows, correlations = [], []
    directional_overlaps = _directional_trade_overlap(validation_trades)
    pairwise = _pairwise_validation_complementarity(
        validation_trades, overlap_rows, correlations, directional_overlaps)
    overlap_by_cell: dict[tuple[str, str, str], int] = {}
    for row in overlap_rows:
        for strategy in (row["strategy_a"], row["strategy_b"]):
            key = (strategy, row["pair"], row["timeframe"])
            overlap_by_cell[key] = overlap_by_cell.get(key, 0) + int(row["overlapping_trade_pairs"])
    for cell in cells:
        key = (cell["strategy_id"], cell["pair"], cell["timeframe"])
        cell["validation_overlapping_trade_pairs_with_other_strategies"] = overlap_by_cell.get(key, 0)
    currency_exposure = _currency_exposure(validation_trades)
    return {
        "cell_count": len(cells), "hypothesis_family_size": len(inference_rows),
        "cost_scenarios": [
            {"name": scenario.name, "spread_round_trip_pips_assumed": scenario.spread_round_trip_pips,
             "commission_round_trip_pips_assumed": scenario.commission_round_trip_pips,
             "slippage_round_trip_pips_assumed": scenario.slippage_round_trip_pips,
             "total_round_trip_pips_assumed": scenario.spread_round_trip_pips
             + scenario.commission_round_trip_pips + scenario.slippage_round_trip_pips,
             "broker_calibrated": False}
            for scenario in scenario_list
        ],
        "primary_statistical_scenario": STRESS_COST_SCENARIO,
        "training_validation_metrics_by_cost": metric_rows,
        "stratified_metrics_at_primary_cost": stratified_rows,
        "cell_evidence": cells,
        "statistical_inference": inference_rows,
        "classification_counts": classifications,
        "holm_correction": {
            "method": "Holm step-down family-wise error control",
            "family_size": len(inference_rows), "alpha": 0.05,
            "family": "all Mission 8 strategy × pair × timeframe validation cells",
            "insufficient_sample_p_value": 1.0,
        },
        "bootstrap_method": {
            "method": "circular moving-block bootstrap on chronologically ordered net trade R outcomes",
            "block_length_trades": BLOCK_LENGTH_TRADES,
            "resamples": n_resamples, "ci": "pointwise 95% percentile interval (not multiplicity-adjusted)",
            "test": "one-sided greater-than-zero centered-null block-bootstrap p-value",
            "familywise_adjustment": "Holm across all 294 validation cells",
        },
        "minimum_evidence_policy": {
            "minimum_validation_trades": MIN_VALIDATION_TRADES,
            "validation_subperiods": VALIDATION_SUBPERIODS,
            "minimum_trades_each_subperiod": MIN_SUBPERIOD_TRADES,
            "stable_status_rule": "positive full-validation mean net R, at least five trades in each of three equal-duration validation subperiods, and positive mean net R in at least two subperiods",
            "validation_candidate_rule": "stable status, at least 25 training trades with positive training mean net R at the same cost scenario, and Holm-adjusted one-sided bootstrap p-value <= 0.05",
        },
        "trade_frequency_policy": (
            "Reported as scoped completed trades per 30 calendar days. Raw signal frequency is unavailable because Mission 8 did not persist split-scoped signal-event counts."
        ),
        "validation_pairwise_complementarity": pairwise,
        "validation_currency_exposure": currency_exposure,
        "potentially_complementary_screened_pairs": [
            row for row in pairwise
            if row["classification"] == "potentially_complementary_requires_signal_event_validation"
        ],
        "complementary_groups_identified": [],
    }


def _pairwise_validation_complementarity(
    trades: pd.DataFrame,
    overlap_rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    directional_overlaps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize only validation trade overlap and validation daily-return correlation."""
    overlap_map = {
        (row["pair"], row["timeframe"], row["strategy_a"], row["strategy_b"]): row
        for row in overlap_rows
    }
    correlation_map = {
        (row["pair"], row["timeframe"], row["strategy_a"], row["strategy_b"]): row
        for row in correlations
    }
    directional_map = {
        (row["pair"], row["timeframe"], row["strategy_a"], row["strategy_b"]): row
        for row in directional_overlaps
    }
    market_cells = sorted(set(zip(trades.pair, trades.timeframe))) if len(trades) else []
    by_strategy_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair, timeframe in market_cells:
        present = sorted(trades.loc[(trades.pair == pair) & (trades.timeframe == timeframe), "strategy_id"].unique())
        for index, first in enumerate(present):
            for second in present[index + 1:]:
                key = (pair, timeframe, first, second)
                overlap = overlap_map.get(key)
                correlation = correlation_map.get(key)
                direction_overlap = directional_map.get(key)
                count_small = min(overlap["trades_a"], overlap["trades_b"]) if overlap else 0
                values = {
                    "pair": pair, "timeframe": timeframe, "strategy_a": first, "strategy_b": second,
                    "overlapping_trade_pairs": int(overlap["overlapping_trade_pairs"]) if overlap else None,
                    "overlap_share_of_smaller_trade_count": (
                        float(overlap["overlapping_trade_pairs"] / count_small) if overlap and count_small else None),
                    "same_direction_overlapping_trade_pairs": (
                        int(direction_overlap["same_direction_overlapping_trade_pairs"])
                        if direction_overlap else None),
                    "opposing_direction_overlapping_trade_pairs": (
                        int(direction_overlap["opposing_direction_overlapping_trade_pairs"])
                        if direction_overlap else None),
                    "validation_daily_exit_return_correlation": (
                        float(correlation["daily_exit_return_correlation"])
                        if correlation and correlation.get("daily_exit_return_correlation") is not None else None),
                    "observed_exit_days": int(correlation["observed_exit_days"]) if correlation else None,
                    "currency_pair": pair, "shared_currency_components": list(pair[:3] + pair[3:]),
                    "signal_overlap_evidence": "not_available: saved Mission 8 signal relationships are not split-scoped",
                }
                by_strategy_pair.setdefault((first, second), []).append(values)
    output = []
    for (first, second), rows in sorted(by_strategy_pair.items()):
        overlap_values = [row["overlap_share_of_smaller_trade_count"] for row in rows
                          if row["overlap_share_of_smaller_trade_count"] is not None]
        correlation_values = [row["validation_daily_exit_return_correlation"] for row in rows
                              if row["validation_daily_exit_return_correlation"] is not None]
        sufficient = len(correlation_values) >= 3 and len(overlap_values) >= 3
        low_corr = sufficient and float(np.median(np.abs(correlation_values))) < 0.3
        low_overlap = sufficient and float(np.median(overlap_values)) < 0.1
        classification = ("potentially_complementary_requires_signal_event_validation"
                          if low_corr and low_overlap else "unknown_or_insufficient_complementarity")
        output.append({
            "strategy_a": first, "strategy_b": second,
            "market_cells_with_both_strategies": len(rows),
            "market_cells_with_known_validation_correlation": len(correlation_values),
            "market_cells_with_overlap_evidence": len(overlap_values),
            "median_absolute_validation_correlation": (
                float(np.median(np.abs(correlation_values))) if correlation_values else None),
            "median_validation_trade_overlap_share": float(np.median(overlap_values)) if overlap_values else None,
            "signal_directional_overlap_status": "unknown: no validation-split signal event log is persisted",
            "same_or_opposing_trade_directions": (
                "counts of overlapping executed trade intervals; not raw signal-event overlap"),
            "shared_currency_exposure": "strategies trade the same listed pair in each included market cell; exposure direction depends on their trade directions",
            "classification": classification,
            "interpretation": "descriptive validation evidence only; not proof of distinct predictive information",
            "market_rows": rows,
        })
    return output


def _directional_trade_overlap(trades: pd.DataFrame) -> list[dict[str, Any]]:
    """Count overlapping validation trade intervals by matching/opposing side."""
    output = []
    for (pair, timeframe), market in trades.groupby(["pair", "timeframe"], sort=True):
        strategies = sorted(market.strategy_id.unique())
        grouped = {
            strategy: list(group.sort_values("entry_timestamp", kind="stable").itertuples(index=False))
            for strategy, group in market.groupby("strategy_id", sort=False)
        }
        for index, first in enumerate(strategies):
            for second in strategies[index + 1:]:
                a, b = grouped[first], grouped[second]
                i = j = same = opposing = 0
                while i < len(a) and j < len(b):
                    left, right = a[i], b[j]
                    overlap = (left.entry_timestamp <= right.exit_timestamp
                               and right.entry_timestamp <= left.exit_timestamp)
                    if overlap:
                        if str(left.direction).upper() == str(right.direction).upper():
                            same += 1
                        else:
                            opposing += 1
                    if left.exit_timestamp < right.exit_timestamp:
                        i += 1
                    else:
                        j += 1
                output.append({
                    "pair": pair, "timeframe": timeframe,
                    "strategy_a": first, "strategy_b": second,
                    "same_direction_overlapping_trade_pairs": same,
                    "opposing_direction_overlapping_trade_pairs": opposing,
                    "overlapping_trade_pairs": same + opposing,
                })
    return output


def _currency_exposure(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    work = trades.copy()
    work["side"] = np.where(work.direction.str.upper() == "BUY", 1, -1)
    work["base_currency"] = work.pair.str[:3]
    work["quote_currency"] = work.pair.str[3:6]
    records = []
    for (strategy, currency), group in pd.concat([
        work[["strategy_id", "base_currency", "side"]].rename(columns={"base_currency": "currency"}).assign(exposure=lambda x: x.side),
        work[["strategy_id", "quote_currency", "side"]].rename(columns={"quote_currency": "currency"}).assign(exposure=lambda x: -x.side),
    ], ignore_index=True).groupby(["strategy_id", "currency"], sort=True):
        records.append({
            "strategy_id": strategy, "currency": currency,
            "validation_trade_count": int(len(group)),
            "signed_trade_count_exposure": int(group.exposure.sum()),
            "unit_notional_proxy_only": True,
            "not_a_broker_lot_or_account_risk_measure": True,
        })
    return records


def _markdown(report: Mapping[str, Any]) -> str:
    counts = report["results"]["classification_counts"]
    lines = [
        "# Mission 13.3 — Pair/Timeframe Evidence & Statistical Validation", "",
        f"- Run: `{report['run_id']}`",
        f"- Generated UTC: `{report['generated_at_utc']}`",
        "- Evidence scope: Mission 8 saved trades; training and validation only.",
        "- Post-validation trade outcomes: skipped; no Mission 12 test data or metrics used.",
        f"- Exact strategy/pair/timeframe cells: {report['results']['cell_count']}",
        f"- Primary cost sensitivity: `{report['results']['primary_statistical_scenario']}` "
        "(3.5 assumed round-trip pips; not broker-calibrated).", "",
        "## Executive summary", "",
        f"- Classification counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Holm-significant cells: {report['summary']['holm_significant_cells']}",
        f"- Potentially complementary strategy pairs: {report['summary']['complementary_pair_count']}",
        "- No validation candidate is automatic approval or a live-trading authorization.", "",
        "## Methodology and assumptions", "",
        f"- Bootstrap: {report['results']['bootstrap_method']['method']}; "
        f"{report['results']['bootstrap_method']['resamples']} resamples; "
        f"block length {report['results']['bootstrap_method']['block_length_trades']} trades.",
        f"- Multiple testing: {report['results']['holm_correction']['method']} across "
        f"{report['results']['holm_correction']['family_size']} exact cells.",
        f"- Stability: {report['results']['minimum_evidence_policy']['stable_status_rule']}.",
        "- The Mission 4 minute-series HAC and circular-shift null test were not applied to irregular trade sequences.",
        "- The current Mission 9 validation period has already been examined in prior research; p-values are exploratory, not fresh confirmation.",
        "- Spread, commission, and slippage are sensitivity assumptions. Commission and slippage are not calibrated; Mission 8 used single-price OHLC fills.", "",
        "## Results for all 14 strategies", "",
        "| Strategy | Cells | Insufficient | Negative | Promising | Validation candidates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGY_IDS:
        group = [cell for cell in report["results"]["cell_evidence"] if cell["strategy_id"] == strategy]
        c = {name: sum(row["classification"] == name for row in group) for name in counts}
        lines.append(
            f"| {strategy} | {len(group)} | {c.get('INSUFFICIENT EVIDENCE', 0)} | "
            f"{c.get('NEGATIVE EVIDENCE', 0)} | {c.get('PROMISING — REQUIRES FURTHER VALIDATION', 0)} | "
            f"{c.get('VALIDATION CANDIDATE', 0)} |"
        )
    lines.extend([
        "", "## Data and research limitations", "",
        *[f"- {item}" for item in report["limitations"]],
        "", "Detailed exact-scope, regime/session/direction, cost, uncertainty, and complementarity rows are in the JSON/CSV artifacts.", "",
    ])
    return "\n".join(lines)


def run_mission13_3(
    *,
    mission8_path: Path = DEFAULT_MISSION8,
    trade_path: Path = DEFAULT_TRADES,
    mission9_path: Path = DEFAULT_MISSION9,
    mission11_path: Path = DEFAULT_MISSION11,
    data_dir: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
    report_stamp: str | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Create immutable, timestamped Mission 13.3 JSON, CSV, and Markdown outputs."""
    for path in (mission8_path, trade_path, mission9_path, mission11_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Mission 13.3 input is missing: {path}")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    payload = json.loads(mission8_path.read_text(encoding="utf-8"))
    if len(payload.get("results", [])) != EXPECTED_FAMILY_SIZE:
        raise ValueError(f"expected {EXPECTED_FAMILY_SIZE} Mission 8 strategy/market rows")
    if payload.get("strategies") != len(STRATEGY_IDS):
        raise ValueError("Mission 8 source does not preserve all 14 strategy engines")
    source_strategy_ids = {row.get("strategy_id") for row in payload["results"]}
    if source_strategy_ids != set(STRATEGY_IDS):
        raise ValueError("Mission 8 result rows do not match all 14 current strategy IDs")
    boundaries = load_split_boundaries(payload, data_dir=data_dir)
    trades, trade_accounting = load_development_trades(trade_path, boundaries)
    validate_saved_split_counts(payload["results"], trades)
    results = evaluate_exact_scope(payload["results"], trades, boundaries, n_resamples=n_resamples)
    stamp = report_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("report_stamp must use YYYYMMDDTHHMMSSZ") from exc
    generated_at = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mission13_3_pair_timeframe_validation_{stamp}"
    paths = {
        "json": output_dir / f"{stem}.json",
        "markdown": output_dir / f"{stem}.md",
        "metrics_csv": output_dir / f"{stem}_metrics.csv",
        "stratified_csv": output_dir / f"{stem}_stratified.csv",
        "complementarity_csv": output_dir / f"{stem}_complementarity.csv",
        "gate_evidence_json": output_dir / f"mission13_3_eligibility_{stamp}.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Mission 13.3 output already exists for run {stamp}")
    boundaries_report = {
        f"{pair}_{timeframe}": {
            "training_start_utc": values[0].isoformat(),
            "validation_start_utc": values[1].isoformat(),
            "validation_end_utc": values[2].isoformat(),
            "source_end_utc": values[3].isoformat(),
            "policy": "chronological 60/20/20 by research-frame timestamp row count; validation end excludes test",
        }
        for (pair, timeframe), values in sorted(boundaries.items())
    }
    boundary_sources = {}
    for item in payload.get("data_exclusions_and_provenance", []):
        source = _resolve_source(item, data_dir)
        pair, timeframe = source.stem.rsplit("_", 1)
        boundary_sources[f"{pair}_{timeframe}"] = str(source.resolve())
    source_files = {
        "mission8_json": str(mission8_path.resolve()), "mission8_trade_log": str(trade_path.resolve()),
        "mission9_cost_report_reference": str(mission9_path.resolve()),
        "mission11_validation_correlation_reference": str(mission11_path.resolve()),
        "boundary_candle_sources": boundary_sources,
        "analysis_code": {
            "mission13_3_evaluator": str(Path(__file__).resolve()),
            "mission9_cost_methods": str(Path(mission9.__file__).resolve()),
            "multi_strategy_engine": str(Path(multi_strategy_engine.__file__).resolve()),
        },
    }
    source_fingerprints = {
        key: _sha256(Path(value))
        for key, value in source_files.items()
        if isinstance(value, str)
    }
    source_fingerprints.update({
        f"boundary_candle_{key}": _sha256(Path(value))
        for key, value in boundary_sources.items()
    })
    source_fingerprints.update({
        f"analysis_code_{key}": _sha256(Path(value))
        for key, value in source_files["analysis_code"].items()
    })
    report = {
        "report_id": "mission13_3_pair_timeframe_validation_v1",
        "run_id": stamp,
        "generated_at_utc": generated_at,
        "source_files": source_files,
        "source_fingerprints_sha256": source_fingerprints,
        "source_data_fingerprints": payload.get("data_fingerprints", {}),
        "dataset_boundaries": boundaries_report,
        "trade_sample_accounting": trade_accounting,
        "results": results,
        "summary": {
            "exact_pair_timeframe_cells": results["cell_count"],
            "sufficient_validation_cells": sum(
                row["classification"] != "INSUFFICIENT EVIDENCE"
                for row in results["cell_evidence"]),
            "holm_significant_cells": sum(
                row["holm_significant_at_0_05"] for row in results["statistical_inference"]),
            "complementary_pair_count": len(results["complementary_groups_identified"]),
            "potentially_complementary_screened_pair_count": len(
                results["potentially_complementary_screened_pairs"]),
        },
        "software_provenance": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "timezone": "UTC", "orders_submitted": False, "broker_access": False,
            "strategies_regenerated": False, "test_outcomes_used": False,
        },
        "limitations": [
            "Mission 8 saved results derive from single-price OHLC fills; costs are fixed round-trip sensitivities, not replayed executable bid/ask fills.",
            "Commission and slippage are uncalibrated. None of the cost scenarios represents confirmed broker costs.",
            "Mission 9 validation was previously examined; this analysis is exploratory and cannot serve as untouched confirmation.",
            "Post-validation trade outcomes are not parsed into the analysis. Existing Mission 12 test results remain untouched.",
            "Saved Mission 8 signal relationship summaries are not split-scoped, so validation signal-event overlap/agreement is unknown.",
            "Daily exit-return correlation is sparse, descriptive, and zero-filled on days with no exits; it does not prove independence.",
            "Currency exposure counts are unit-trade proxies, not account-currency notional, lot sizing, or portfolio risk.",
            "Moving-block bootstrap over trade order is an approximate dependence-aware method; intervals are pointwise and not multiplicity-adjusted.",
            "A validation candidate requires genuinely new chronological confirmation before any strategy eligibility decision.",
        ],
        "report_files": {key: str(value.resolve()) for key, value in paths.items()},
    }
    gate_payload = {
        "report_id": report["report_id"], "run_id": stamp,
        "generated_at_utc": generated_at,
        "source_fingerprints_sha256": source_fingerprints,
        "primary_statistical_scenario": STRESS_COST_SCENARIO,
        "cells": [
            {key: row[key] for key in (
                "strategy_id", "pair", "timeframe", "pair_timeframe_stability_status",
                "classification", "validation_metrics", "validation_periods",
            )}
            for row in results["cell_evidence"]
        ],
    }
    flat = json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    gate_text = json.dumps(_jsonable(gate_payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    content = {
        paths["json"]: flat,
        paths["markdown"]: _markdown(report),
        paths["metrics_csv"]: pd.DataFrame(results["training_validation_metrics_by_cost"]).to_csv(index=False),
        paths["stratified_csv"]: pd.DataFrame(results["stratified_metrics_at_primary_cost"]).to_csv(index=False),
        paths["complementarity_csv"]: pd.DataFrame([
            {key: value for key, value in row.items() if key != "market_rows"}
            for row in results["validation_pairwise_complementarity"]
        ]).to_csv(index=False),
        paths["gate_evidence_json"]: gate_text,
    }
    created = []
    try:
        for path, text in content.items():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            created.append(path)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission8-json", type=Path, default=DEFAULT_MISSION8)
    parser.add_argument("--trade-log", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--mission9-json", type=Path, default=DEFAULT_MISSION9)
    parser.add_argument("--mission11-json", type=Path, default=DEFAULT_MISSION11)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-stamp", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    report = run_mission13_3(
        mission8_path=args.mission8_json, trade_path=args.trade_log,
        mission9_path=args.mission9_json, mission11_path=args.mission11_json,
        data_dir=args.data_dir, output_dir=args.output_dir,
        report_stamp=args.report_stamp, n_resamples=args.bootstrap_resamples,
    )
    print(json.dumps({"summary": report["summary"], "report_files": report["report_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
