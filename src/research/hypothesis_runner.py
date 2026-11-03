"""Registered production entry point and private hypothesis evaluator.

This module reuses the shared compiler, split helper, attribution checks, and
sequential OHLC backtester. It does not persist runs or integrate orchestration.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pandas as pd

from .backtester import run_backtest
from .experiment_ema import split_metadata, timeline_periods, validate_signal_attribution
from .hypothesis_schema import StrategyHypothesis
from .hypothesis_signals import (
    SignalCompilation, compile_hypothesis_signals, prepare_hypothesis_market_data,
)
from .hypothesis_validator import validate_hypothesis
from .data import load_market_data, normalize_utc_timestamps, validate_m1_cadence
from .execution_models import (
    BidAskExecutionModel,
    MarketDataMetadata,
    prepare_bid_ask_inputs,
)


def _evaluate_period(
    compilation: SignalCompilation,
    hypothesis: StrategyHypothesis,
    period: dict[str, Any],
    *,
    ask_market_data: pd.DataFrame | None = None,
    execution_model: BidAskExecutionModel | None = None,
    bid_metadata: MarketDataMetadata | None = None,
    ask_metadata: MarketDataMetadata | None = None,
) -> dict[str, Any]:
    market = compilation.market_data
    signals = compilation.signals
    start, end = period["start"], period["end"]
    period_mask = (signals["decision_timestamp"] >= start) & (signals["decision_timestamp"] < end)
    candidates = signals.loc[period_mask].copy()
    candidate_verified, candidate_unavailable = validate_signal_attribution(
        candidates, market, expected_signal_id_prefix=compilation.signal_id_prefix,
    )

    warm = candidates["indicator_warmup_complete"].astype(bool)
    warmup_rejected = candidates.loc[~warm].copy()
    if not warmup_rejected.empty:
        warmup_rejected["rejection_reason"] = "indicator_warmup_incomplete"
    ready_signals = candidates.loc[warm].copy()

    execution = hypothesis.execution_assumptions
    bar = pd.Timedelta(execution.bar_duration)
    if execution.timestamp_convention == "bar_open":
        candle_end = market["time"] + bar
    else:
        candle_end = market["time"]
    available_market = market.loc[candle_end < end].copy()
    backtest_columns = [
        "decision_timestamp", "direction", "signal_id", "source_row_index",
        "source_candle_timestamp",
    ]
    backtest_kwargs = {
        "holding_period": hypothesis.exit_rule.max_holding_period,
        "timestamp_convention": execution.timestamp_convention,
        "execution_delay": execution.execution_delay,
        "allow_overlapping": execution.allow_overlapping,
        "transaction_cost_pips": execution.transaction_cost_pips,
        "pip_size": execution.pip_size,
        "bar_duration": execution.bar_duration,
        "gap_tolerance": execution.gap_tolerance,
    }
    if execution_model is not None:
        backtest_kwargs.update({
            "transaction_cost_pips": 0.0,
            "execution_model": execution_model,
            "ask_data": ask_market_data,
            "bid_metadata": bid_metadata,
            "ask_metadata": ask_metadata,
        })
    result = run_backtest(ready_signals[backtest_columns], available_market, **backtest_kwargs)
    accepted_verified, accepted_unavailable = validate_signal_attribution(
        result["trades"], market, expected_signal_id_prefix=compilation.signal_id_prefix,
    )
    rejected_signals = pd.concat(
        [warmup_rejected, result["rejected_signals"]], ignore_index=True, sort=False,
    )
    rejected_verified, rejected_unavailable = validate_signal_attribution(
        rejected_signals, market, expected_signal_id_prefix=compilation.signal_id_prefix,
    )

    candidate_count = len(candidates)
    warmup_count = len(warmup_rejected)
    backtester_input_count = len(ready_signals)
    backtester_rejected_count = sum(result["rejected"].values())
    accepted_count = len(result["trades"])
    if candidate_count != warmup_count + backtester_input_count:
        raise AssertionError("candidate and warm-up accounting does not reconcile")
    if backtester_input_count != accepted_count + backtester_rejected_count:
        raise AssertionError("backtester accepted/rejected accounting does not reconcile")
    if not result["trades"].empty and not (result["trades"]["exit_timestamp"] < end).all():
        raise AssertionError("trade outcome crosses its assigned split boundary")

    return {
        **result,
        "status": "evaluated",
        "confirmatory": False,
        "selection_eligible": True,
        "period_start_utc": start,
        "period_end_utc": end,
        "candidate_events": candidate_count,
        "warmup_excluded": warmup_count,
        "backtester_inputs": backtester_input_count,
        "backtester_rejected": backtester_rejected_count,
        "accepted_trades": accepted_count,
        "rejected_signals": rejected_signals,
        "attribution": {
            "candidates_verified": candidate_verified,
            "candidates_unavailable": candidate_unavailable,
            "accepted_verified": accepted_verified,
            "accepted_unavailable": accepted_unavailable,
            "rejected_verified": rejected_verified,
            "rejected_unavailable": rejected_unavailable,
        },
        "data_quality": result["market_data_report"],
    }


def _evaluate_hypothesis_unregistered(
    hypothesis: StrategyHypothesis,
    market_data: pd.DataFrame | None = None,
    *,
    execution_model: BidAskExecutionModel | None = None,
    ask_market_data: pd.DataFrame | None = None,
    bid_metadata: MarketDataMetadata | None = None,
    ask_metadata: MarketDataMetadata | None = None,
) -> dict[str, Any]:
    """Internal evaluation primitive; callers must use the registered API.

    Period boundaries use the full ordered candle timeline and the existing
    60/20/20 split. Outcomes are censored so exits remain strictly inside their
    assigned period. The previously accessed test interval is recorded as
    ``not_evaluated`` and cannot be selected. This private function exists for
    deterministic synthetic unit fixtures; production execution must use
    :func:`run_registered_hypothesis`.
    """
    validation = validate_hypothesis(hypothesis)
    execution = hypothesis.execution_assumptions
    if pd.Timedelta(hypothesis.execution_assumptions.bar_duration) != pd.Timedelta("1min"):
        raise ValueError("the current hypothesis runner supports M1 bar_duration only")
    raw_market = load_market_data() if market_data is None else market_data.copy()
    if "time" not in raw_market:
        raise ValueError("market_data must contain time")
    raw_times = normalize_utc_timestamps(raw_market["time"])
    if raw_times.isna().any() or raw_times.duplicated().any() or not raw_times.is_monotonic_increasing:
        raise ValueError("market timestamps must be valid, ordered, and unique")
    periods = timeline_periods(pd.DataFrame({"time": raw_times}),
                               bar_duration=hypothesis.execution_assumptions.bar_duration)
    # The complete timestamp index defines the frozen split. OHLC after the old
    # test boundary is then removed before validation or feature construction.
    development_mask = raw_times < periods["test"]["start"]
    raw_market = raw_market.loc[development_mask].copy()
    raw_market["time"] = raw_times.loc[development_mask]
    market, _ = prepare_hypothesis_market_data(raw_market)
    cadence = validate_m1_cadence(market["time"])
    pair_report = None
    if execution_model is None:
        if ask_market_data is not None or bid_metadata is not None or ask_metadata is not None:
            raise ValueError("BID/ASK data requires an explicit side-aware execution model")
    else:
        if execution.transaction_cost_pips != 0.0:
            raise ValueError(
                "registered BID/ASK hypotheses must set transaction_cost_pips to zero; "
                "spread is embedded in side-specific prices"
            )
        if ask_market_data is None or bid_metadata is None or ask_metadata is None:
            raise ValueError("BID/ASK execution requires ASK data and both side metadata records")
        raw_ask = ask_market_data.copy()
        if "time" not in raw_ask:
            raise ValueError("ask_market_data must contain time")
        ask_times = normalize_utc_timestamps(raw_ask["time"])
        if ask_times.isna().any() or ask_times.duplicated().any() or not ask_times.is_monotonic_increasing:
            raise ValueError("ASK timestamps must be valid, ordered, and unique")
        ask_mask = ask_times < periods["test"]["start"]
        raw_ask = raw_ask.loc[ask_mask].copy()
        raw_ask["time"] = ask_times.loc[ask_mask]
        market, ask_market_data, pair_report = prepare_bid_ask_inputs(
            market, raw_ask, bid_metadata=bid_metadata, ask_metadata=ask_metadata,
            timestamp_convention=execution.timestamp_convention,
            bar_duration=execution.bar_duration,
        )
    compilation = compile_hypothesis_signals(hypothesis, market)
    # Keep test-period candidate signals out of the runner's returned object;
    # only train and validation events are evaluated or summarized here.
    development_mask = compilation.signals["decision_timestamp"] < periods["test"]["start"]
    development_signals = compilation.signals.loc[development_mask].copy()
    compilation = replace(
        compilation,
        signals=development_signals,
        candidate_events=len(development_signals),
        warmup_suppressed=int((~development_signals["indicator_warmup_complete"].astype(bool)).sum()),
    )
    split = split_metadata(periods)
    period_results = {
        period_name: _evaluate_period(
            compilation, hypothesis, periods[period_name], ask_market_data=ask_market_data,
            execution_model=execution_model, bid_metadata=bid_metadata, ask_metadata=ask_metadata,
        )
        for period_name in ("train", "validation")
    }
    period_results["test"] = {
        "status": "not_evaluated",
        "confirmatory": False,
        "selection_eligible": False,
        "previously_inspected": True,
        "period_start_utc": periods["test"]["start"],
        "period_end_utc": periods["test"]["end"],
        "metrics": None,
        "reason": "Test evaluation is intentionally disabled for this isolated runner.",
    }
    evaluated = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "hypothesis_version": hypothesis.version,
        "semantic_digest": validation.semantic_digest,
        "minimum_warmup_bars": validation.minimum_warmup_bars,
        "candidate_events_total": compilation.candidate_events,
        "warmup_suppressed_total": compilation.warmup_suppressed,
        "data_quality": compilation.data_quality,
        "cadence": cadence,
        "split_metadata": split,
        "periods": period_results,
    }
    if execution_model is not None:
        evaluated["execution_model"] = {
            "identifier": execution_model.identifier,
            "version": execution_model.version,
            "alignment_policy": execution_model.alignment_policy,
            "alignment_report": pair_report,
            "commission_pips_round_trip": execution_model.commission_pips_round_trip,
            "slippage_pips_round_trip": execution_model.slippage_pips_round_trip,
            "commission_and_slippage_are_assumptions": True,
            "legacy_fixed_round_trip_cost_applied": False,
        }
    return evaluated


def run_registered_hypothesis(
    hypothesis_id: str,
    version: str,
    *,
    store: Any | None = None,
    store_path: str | None = None,
    market_data: pd.DataFrame | None = None,
    execution_model: BidAskExecutionModel | None = None,
    ask_market_data: pd.DataFrame | None = None,
    bid_metadata: MarketDataMetadata | None = None,
    ask_metadata: MarketDataMetadata | None = None,
) -> dict[str, Any]:
    """Load, verify, then evaluate the exact explicitly registered hypothesis.

    This is the production execution boundary. Unregistered proposals and
    callers' in-memory substitutes are rejected before any signal compilation.
    """
    from pathlib import Path

    from .experiment_store import DEFAULT_DB_PATH, ExperimentStore
    from .hypothesis_schema import StrategyHypothesis

    if store is None:
        database = Path(store_path) if store_path is not None else DEFAULT_DB_PATH
        store = ExperimentStore(database, initialize=False)
    record = store.get_registered_hypothesis(hypothesis_id, version)
    if record.get("validation_status") != "valid" or record.get("registration_status") != "registered":
        raise ValueError("hypothesis is not valid and explicitly registered")
    hypothesis = StrategyHypothesis.from_dict(record["hypothesis"])
    validation = validate_hypothesis(hypothesis)
    if (hypothesis.hypothesis_id != hypothesis_id or hypothesis.version != version
            or validation.semantic_digest != record.get("semantic_digest")
            or hypothesis.canonical_behavior() != record.get("canonical_config")):
        raise ValueError("stored registered hypothesis identity or configuration is inconsistent")
    return _evaluate_hypothesis_unregistered(
        hypothesis, market_data, execution_model=execution_model,
        ask_market_data=ask_market_data, bid_metadata=bid_metadata, ask_metadata=ask_metadata,
    )


def run_registered_hypothesis_windows(
    hypothesis_id: str,
    version: str,
    windows: list[Any] | tuple[Any, ...],
    *,
    execution_models: list[BidAskExecutionModel] | tuple[BidAskExecutionModel, ...],
    store: Any | None = None,
    store_path: str | None = None,
    market_data: pd.DataFrame | None = None,
    ask_market_data: pd.DataFrame | None = None,
    bid_metadata: MarketDataMetadata | None = None,
    ask_metadata: MarketDataMetadata | None = None,
) -> dict[str, Any]:
    """Evaluate exact registered rules in caller-supplied windows within train only.

    Timeline partitioning reads timestamps for the full ordered dataset, then
    slices BID/ASK rows at the outer training boundary before OHLC validation,
    feature construction, signal generation, or backtesting. The test interval
    is never feature-built or sent to the backtester.
    """
    from pathlib import Path

    from .experiment_store import DEFAULT_DB_PATH, ExperimentStore
    from .hypothesis_schema import StrategyHypothesis

    if not windows:
        raise ValueError("at least one fold window is required")
    if not execution_models or any(not isinstance(model, BidAskExecutionModel) for model in execution_models):
        raise ValueError("one or more explicit BID/ASK execution models are required")
    if store is None:
        database = Path(store_path) if store_path is not None else DEFAULT_DB_PATH
        store = ExperimentStore(database, initialize=False)
    record = store.get_registered_hypothesis(hypothesis_id, version)
    if record.get("validation_status") != "valid" or record.get("registration_status") != "registered":
        raise ValueError("hypothesis is not valid and explicitly registered")
    hypothesis = StrategyHypothesis.from_dict(record["hypothesis"])
    validation = validate_hypothesis(hypothesis)
    if (hypothesis.hypothesis_id != hypothesis_id or hypothesis.version != version
            or validation.semantic_digest != record.get("semantic_digest")
            or hypothesis.canonical_behavior() != record.get("canonical_config")):
        raise ValueError("stored registered hypothesis identity or configuration is inconsistent")
    if pd.Timedelta(hypothesis.execution_assumptions.bar_duration) != pd.Timedelta("1min"):
        raise ValueError("walk-forward hypothesis execution supports M1 only")
    if hypothesis.execution_assumptions.transaction_cost_pips != 0.0:
        raise ValueError("BID/ASK registered hypothesis must not apply legacy fixed cost")
    if market_data is None:
        raw = load_market_data()
    elif isinstance(market_data, pd.DataFrame):
        raw = market_data.copy()
    else:
        raise TypeError("market_data must be a pandas DataFrame")
    if "time" not in raw:
        raise ValueError("market_data must contain time")
    raw_times = normalize_utc_timestamps(raw["time"])
    if raw_times.isna().any() or raw_times.duplicated().any() or not raw_times.is_monotonic_increasing:
        raise ValueError("timeline timestamps must be valid, ordered, and unique")
    timeline = pd.DataFrame({"time": raw_times})
    boundaries = timeline_periods(timeline, bar_duration=hypothesis.execution_assumptions.bar_duration)
    train_end = boundaries["train"]["end"]
    training_raw = raw.loc[raw_times < train_end].copy()
    training_raw["time"] = raw_times.loc[raw_times < train_end]
    training_market, _ = prepare_hypothesis_market_data(training_raw)
    cadence = validate_m1_cadence(training_market["time"])
    if ask_market_data is None or bid_metadata is None or ask_metadata is None:
        raise ValueError("walk-forward BID/ASK evaluation requires ASK data and metadata for both sides")
    ask_raw = ask_market_data.copy()
    if "time" not in ask_raw:
        raise ValueError("ask_market_data must contain time")
    ask_times = normalize_utc_timestamps(ask_raw["time"])
    if ask_times.isna().any() or ask_times.duplicated().any() or not ask_times.is_monotonic_increasing:
        raise ValueError("ASK timeline timestamps must be valid, ordered, and unique")
    ask_training = ask_raw.loc[ask_times < train_end].copy()
    ask_training["time"] = ask_times.loc[ask_times < train_end]
    training_market, ask_training, alignment = prepare_bid_ask_inputs(
        training_market, ask_training, bid_metadata=bid_metadata, ask_metadata=ask_metadata,
        timestamp_convention=hypothesis.execution_assumptions.timestamp_convention,
        bar_duration=hypothesis.execution_assumptions.bar_duration,
    )
    compilation = compile_hypothesis_signals(hypothesis, training_market)
    folds: list[dict[str, Any]] = []
    for fold in windows:
        fold_values = asdict(fold) if hasattr(fold, "__dataclass_fields__") else dict(fold)
        start = pd.Timestamp(fold_values["evaluation_start_utc"])
        end = pd.Timestamp(fold_values["evaluation_end_utc"])
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("fold boundaries must be timezone-aware")
        start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
        if start >= end or start < boundaries["train"]["start"] or end > train_end:
            raise ValueError("all evaluation folds must lie wholly inside the outer training period")
        per_scenario: dict[str, Any] = {}
        for model in execution_models:
            key = f"commission_{model.commission_pips_round_trip:g}_slippage_{model.slippage_pips_round_trip:g}"
            result = _evaluate_period(
                compilation, hypothesis, {"start": start, "end": end},
                ask_market_data=ask_training, execution_model=model,
                bid_metadata=bid_metadata, ask_metadata=ask_metadata,
            )
            per_scenario[key] = result
        folds.append({**fold_values, "scenarios": per_scenario})
    return {
        "hypothesis_id": hypothesis_id, "version": version,
        "semantic_digest": validation.semantic_digest,
        "outer_training_period": {
            "start_utc": boundaries["train"]["start"].isoformat(),
            "end_utc": train_end.isoformat(), "rows": int(len(training_market)),
        },
        "training_cadence": cadence, "bid_ask_alignment": alignment,
        "candidate_events_in_training": int(len(compilation.signals)),
        "folds": folds,
        "validation": {"status": "not_evaluated_during_training_screen",
                       "selection_eligible": False, "test": "not read by feature builder or backtester"},
    }
