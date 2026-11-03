"""Causal compiler for validator-approved declarative hypothesis rules.

This compiler only evaluates the restricted rule-tree operators. It contains no
expression evaluation, generated-code execution, labels, or outcome fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import load_market_data, normalize_utc_timestamps, validate_ohlc_prices
from .features import build_features, indicator_segment_ids, segment_bar_counts
from .hypothesis_schema import FeatureRef, RuleNode, StrategyHypothesis
from .hypothesis_validator import (
    BOOLEAN_OPERATORS,
    COMPARISON_OPERATORS,
    CROSSING_OPERATORS,
    SUPPORTED_PRICE_FIELDS,
    validate_hypothesis,
)


@dataclass(frozen=True)
class SignalCompilation:
    signals: pd.DataFrame
    market_data: pd.DataFrame
    signal_id_prefix: str
    candidate_events: int
    warmup_suppressed: int
    minimum_warmup_bars: int
    data_quality: dict[str, Any]


SIGNAL_COLUMNS = [
    "signal_id", "decision_timestamp", "direction", "source_row_index",
    "source_index", "source_candle_timestamp", "indicator_segment_id",
    "segment_bar_count", "indicator_warmup_complete",
]


def _prepare_market_data(data: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    market = load_market_data() if data is None else data.copy()
    load_report = dict(market.attrs.get("load_report", {}))
    input_rows = len(market)
    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(market.columns):
        raise ValueError(f"market_data must include {sorted(required)}")
    market["time"] = normalize_utc_timestamps(market["time"])
    missing_times = market["time"].isna()
    dropped_timestamps = int(missing_times.sum())
    market = market.loc[~missing_times].copy()
    market = validate_ohlc_prices(market)
    duplicate_count = int(market["time"].duplicated(keep="last").sum())
    market = (market.sort_values("time")
              .drop_duplicates("time", keep="last")
              .reset_index(drop=True))
    if market.empty:
        raise ValueError("market_data has no valid candles")
    quality = {
        "input_rows": input_rows,
        "dropped_invalid_timestamps": dropped_timestamps,
        "dropped_duplicate_timestamps": duplicate_count,
        "rows_used": len(market),
        "source_load_report": load_report,
    }
    return market, quality


def prepare_hypothesis_market_data(
    data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the same normalized frame and quality record used by compilation."""
    return _prepare_market_data(data)


def _operand_values(
    operand: Any,
    features: pd.DataFrame,
    segments: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(operand, FeatureRef):
        if operand.name not in SUPPORTED_PRICE_FIELDS and operand.name not in features.columns:
            raise ValueError(f"unsupported feature at compile time: {operand.name!r}")
        values = features[operand.name]
        if operand.offset_bars:
            values = values.groupby(segments, sort=False).shift(operand.offset_bars)
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return numeric, np.isfinite(numeric)
    if isinstance(operand, bool) or not isinstance(operand, (int, float)):
        raise ValueError(f"unsupported rule operand at compile time: {operand!r}")
    try:
        scalar = float(operand)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"unsupported numeric rule operand: {operand!r}") from exc
    if not np.isfinite(scalar):
        raise ValueError("rule constants must be finite")
    return np.full(len(features), scalar, dtype=float), np.ones(len(features), dtype=bool)


def _previous(values: np.ndarray, segments: pd.Series) -> np.ndarray:
    series = pd.Series(values, index=segments.index)
    return series.groupby(segments, sort=False).shift(1).to_numpy(dtype=float)


def _evaluate_rule(
    rule: RuleNode,
    features: pd.DataFrame,
    segments: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (truth values, availability mask) for one validated rule tree."""
    if not isinstance(rule, RuleNode):
        raise ValueError("rule node must be a RuleNode")
    operator = rule.operator
    operands = rule.operands
    if operator in COMPARISON_OPERATORS:
        if len(operands) != 2:
            raise ValueError(f"{operator} requires exactly two operands")
        left, left_valid = _operand_values(operands[0], features, segments)
        right, right_valid = _operand_values(operands[1], features, segments)
        functions = {
            "gt": np.greater, "gte": np.greater_equal, "lt": np.less,
            "lte": np.less_equal, "eq": np.equal, "ne": np.not_equal,
        }
        available = left_valid & right_valid
        return functions[operator](left, right) & available, available

    if operator in CROSSING_OPERATORS:
        if len(operands) != 2:
            raise ValueError(f"{operator} requires exactly two operands")
        left, left_valid = _operand_values(operands[0], features, segments)
        right, right_valid = _operand_values(operands[1], features, segments)
        prior_left, prior_right = _previous(left, segments), _previous(right, segments)
        prior_left_valid = np.isfinite(prior_left)
        prior_right_valid = np.isfinite(prior_right)
        if isinstance(operands[0], (int, float)) and not isinstance(operands[0], bool):
            prior_left = left.copy()
            prior_left_valid[:] = left_valid
        if isinstance(operands[1], (int, float)) and not isinstance(operands[1], bool):
            prior_right = right.copy()
            prior_right_valid[:] = right_valid
        available = left_valid & right_valid & prior_left_valid & prior_right_valid
        if operator == "crosses_above":
            values = (prior_left <= prior_right) & (left > right)
        else:
            values = (prior_left >= prior_right) & (left < right)
        return values & available, available

    if operator in BOOLEAN_OPERATORS:
        if operator == "not":
            if len(operands) != 1:
                raise ValueError("not requires exactly one rule operand")
            child_values, child_available = _evaluate_rule(operands[0], features, segments)
            return ~child_values & child_available, child_available
        if len(operands) < 2:
            raise ValueError(f"{operator} requires at least two rule operands")
        evaluated = [_evaluate_rule(child, features, segments) for child in operands]
        all_available = np.logical_and.reduce([item[1] for item in evaluated])
        if operator == "and":
            values = np.logical_and.reduce([item[0] for item in evaluated])
        else:
            values = np.logical_or.reduce([item[0] for item in evaluated])
        return values & all_available, all_available

    raise ValueError(f"unsupported rule operator at compile time: {operator!r}")


def compile_hypothesis_signals(
    hypothesis: StrategyHypothesis,
    market_data: pd.DataFrame | None = None,
) -> SignalCompilation:
    """Validate and compile one hypothesis into attributed close-time signals.

    Candidate rules are evaluated after the source candle closes. Lag offsets
    and crossing history are segment-local; unavailable values never satisfy a
    rule (including through Boolean negation). Candidates before the declared
    warm-up remain in the output with readiness false so callers can account
    for them, but the runner must not send them to the backtester.
    """
    validation = validate_hypothesis(hypothesis)
    effective_warmup = max(
        hypothesis.required_warmup_bars, validation.minimum_warmup_bars,
    )
    execution = hypothesis.execution_assumptions
    market, quality = _prepare_market_data(market_data)
    features = build_features(market, gap_tolerance=execution.gap_tolerance)
    segments = indicator_segment_ids(features["time"], gap_tolerance=execution.gap_tolerance)
    bars_in_segment = segment_bar_counts(segments)

    long_values, long_available = _evaluate_rule(hypothesis.long_entry, features, segments)
    short_values, short_available = _evaluate_rule(hypothesis.short_entry, features, segments)
    long_signals = long_values & long_available
    short_signals = short_values & short_available
    conflicts = long_signals & short_signals
    if conflicts.any():
        rows = np.flatnonzero(conflicts)[:10].tolist()
        raise ValueError(f"hypothesis produces simultaneous long and short signals at rows {rows}")
    selected = np.flatnonzero(long_signals | short_signals)
    warmup_ready = bars_in_segment.iloc[selected].to_numpy() >= effective_warmup
    selected_times = features["time"].iloc[selected].reset_index(drop=True)
    bar_duration = pd.Timedelta(execution.bar_duration)
    if execution.timestamp_convention == "bar_open":
        decision_times = selected_times + bar_duration
    elif execution.timestamp_convention == "bar_close":
        decision_times = selected_times
    else:  # Validator currently rejects this; retain a compiler-level guard.
        raise ValueError("unsupported timestamp convention at compile time")
    prefix = f"hypothesis_{hypothesis.hypothesis_id}_{validation.semantic_digest[:16]}"

    signals = pd.DataFrame({
        "signal_id": [f"{prefix}:{stamp.isoformat()}" for stamp in selected_times],
        "decision_timestamp": decision_times,
        "direction": np.where(long_signals[selected], "long", "short"),
        "source_row_index": selected,
        "source_index": selected,
        "source_candle_timestamp": selected_times,
        "indicator_segment_id": segments.iloc[selected].to_numpy(),
        "segment_bar_count": bars_in_segment.iloc[selected].to_numpy(),
        "indicator_warmup_complete": warmup_ready,
    }, columns=SIGNAL_COLUMNS)
    candidate_count = len(signals)
    return SignalCompilation(
        signals=signals,
        market_data=market,
        signal_id_prefix=prefix,
        candidate_events=candidate_count,
        warmup_suppressed=int((~warmup_ready).sum()),
        minimum_warmup_bars=effective_warmup,
        data_quality=quality,
    )
