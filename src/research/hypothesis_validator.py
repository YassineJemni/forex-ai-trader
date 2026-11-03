"""Structural validation for declarative strategy hypotheses.

Validation checks representation, causality, available inputs, and whether the
current fixed-duration OHLC backtester can express the execution contract. It
does not load data, calculate outcomes, or assess strategy performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any

import pandas as pd

from .hypothesis_schema import (
    ExecutionAssumptions,
    ExitRule,
    FeatureRef,
    IndicatorInput,
    RuleNode,
    RuleValue,
    StrategyHypothesis,
)


STABLE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
SUPPORTED_FAMILIES = frozenset({
    "trend_following", "mean_reversion", "breakout", "momentum", "volatility_filter",
})
COMPARISON_OPERATORS = frozenset({"gt", "gte", "lt", "lte", "eq", "ne"})
CROSSING_OPERATORS = frozenset({"crosses_above", "crosses_below"})
BOOLEAN_OPERATORS = frozenset({"and", "or", "not"})
SUPPORTED_OPERATORS = COMPARISON_OPERATORS | CROSSING_OPERATORS | BOOLEAN_OPERATORS
SUPPORTED_PRICE_FIELDS = frozenset({"open", "high", "low", "close"})
MAX_OFFSET_BARS = 500


@dataclass(frozen=True)
class IndicatorDefinition:
    """A current pipeline feature constructor and its warm-up convention."""

    parameters: Any
    output_name: str
    minimum_warmup_bars: int


def _definition(name: str, parameters: dict[str, frozenset[Any]], output: str, warmup: int) -> IndicatorDefinition:
    return IndicatorDefinition(MappingProxyType(dict(parameters)), output, warmup)


INDICATORS = MappingProxyType({
    "ema": _definition("ema", {"span": frozenset({9, 21, 50})}, "ema_{span}", 0),
    "ema_gap": _definition("ema_gap", {}, "ema_gap", 105),
    "ema_long_gap": _definition("ema_long_gap", {}, "ema_long_gap", 250),
    "return": _definition("return", {"period": frozenset({1, 5, 15, 60})}, "return_{period}", 0),
    "rsi": _definition("rsi", {"period": frozenset({14})}, "rsi_14", 15),
    "volatility": _definition("volatility", {"period": frozenset({15, 60})}, "volatility_{period}", 0),
    "candle_range": _definition("candle_range", {}, "candle_range", 1),
    "candle_body": _definition("candle_body", {}, "candle_body", 1),
    "time_of_day": _definition("time_of_day", {"component": frozenset({"sin", "cos"})}, "time_{component}", 1),
})


@dataclass(frozen=True)
class ValidationReport:
    minimum_warmup_bars: int
    indicator_warmup_bars: dict[str, int]
    semantic_digest: str


def _positive_duration(value: Any, field: str) -> pd.Timedelta:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a duration string")
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid duration") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError(f"{field} must be positive")
    return duration


def _nonnegative_duration(value: Any, field: str) -> pd.Timedelta:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a duration string")
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid duration") from exc
    if pd.isna(duration) or duration < pd.Timedelta(0):
        raise ValueError(f"{field} cannot be negative")
    return duration


def _validate_execution(execution: Any) -> None:
    if not isinstance(execution, ExecutionAssumptions):
        raise ValueError("execution_assumptions must use the supported ExecutionAssumptions schema")
    if execution.timestamp_convention not in {"bar_open", "bar_close"}:
        raise ValueError("unsupported timestamp_convention")
    _positive_duration(execution.bar_duration, "bar_duration")
    _nonnegative_duration(execution.execution_delay, "execution_delay")
    _nonnegative_duration(execution.gap_tolerance, "gap_tolerance")
    if not isinstance(execution.allow_overlapping, bool):
        raise ValueError("allow_overlapping must be boolean")
    if (isinstance(execution.transaction_cost_pips, bool)
            or not isinstance(execution.transaction_cost_pips, (int, float))
            or not math.isfinite(execution.transaction_cost_pips)
            or execution.transaction_cost_pips < 0):
        raise ValueError("transaction_cost_pips must be finite and nonnegative")
    if (isinstance(execution.pip_size, bool) or not isinstance(execution.pip_size, (int, float))
            or not math.isfinite(execution.pip_size) or execution.pip_size <= 0):
        raise ValueError("pip_size must be finite and positive")


def _indicator_output(indicator: IndicatorInput) -> tuple[str, int]:
    if not isinstance(indicator, IndicatorInput):
        raise ValueError("indicators must contain IndicatorInput values")
    if indicator.name not in INDICATORS:
        raise ValueError(f"unsupported indicator {indicator.name!r}")
    definition = INDICATORS[indicator.name]
    parameters = dict(indicator.parameters)
    if set(parameters) != set(definition.parameters):
        raise ValueError(
            f"{indicator.name} parameters must be exactly {sorted(definition.parameters)}"
        )
    for parameter, allowed in definition.parameters.items():
        value = parameters[parameter]
        if isinstance(value, bool) or value not in allowed:
            raise ValueError(
                f"invalid {indicator.name}.{parameter}={value!r}; supported values: {sorted(allowed)}"
            )
    if indicator.name == "ema":
        span = parameters["span"]
        # Existing feature EMAs are recursive. Five spans are the project's
        # practical stabilization convention, not a mathematical removal of seed influence.
        minimum = 5 * span
    elif indicator.name == "return":
        minimum = parameters["period"] + 1
    elif indicator.name == "volatility":
        # Rolling sample standard deviation of return_1; first return needs two bars.
        minimum = parameters["period"] + 1
    else:
        minimum = definition.minimum_warmup_bars
    output = definition.output_name.format(**parameters)
    return output, minimum


def _walk_rule(node: Any, available_features: set[str]) -> tuple[int, set[str]]:
    if not isinstance(node, RuleNode):
        raise ValueError("entry rules must be RuleNode trees")
    if not isinstance(node.operator, str) or node.operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported rule operator {node.operator!r}")
    operands = node.operands
    operator = node.operator
    if operator in COMPARISON_OPERATORS | CROSSING_OPERATORS:
        if len(operands) != 2:
            raise ValueError(f"{operator} requires exactly two operands")
        if operator in CROSSING_OPERATORS and not all(isinstance(item, (FeatureRef, int, float))
                                                       and not isinstance(item, bool) for item in operands):
            raise ValueError(f"{operator} operands must be feature references or finite numbers")
        minimum, used = 1, set()
        for operand in operands:
            if isinstance(operand, FeatureRef):
                if not isinstance(operand.name, str) or not operand.name:
                    raise ValueError("feature reference name must be nonempty")
                if isinstance(operand.offset_bars, bool) or not isinstance(operand.offset_bars, int):
                    raise ValueError("feature offset_bars must be an integer")
                if operand.offset_bars < 0:
                    raise ValueError("negative offsets reference future data and are forbidden")
                if operand.offset_bars > MAX_OFFSET_BARS:
                    raise ValueError(f"offset_bars exceeds supported maximum {MAX_OFFSET_BARS}")
                if operand.name in {"target", "label", "future_outcome_end_time", "entry_price", "exit_price"}:
                    raise ValueError(f"future/label field {operand.name!r} is forbidden in rules")
                if operand.name not in SUPPORTED_PRICE_FIELDS and operand.name not in available_features:
                    raise ValueError(f"unsupported or unavailable feature {operand.name!r}")
                minimum = max(minimum, operand.offset_bars + 1)
                used.add(operand.name)
            elif (isinstance(operand, bool) or not isinstance(operand, (int, float))
                  or not math.isfinite(operand)):
                raise ValueError(f"{operator} operands must be feature references or finite numbers")
        if operator in CROSSING_OPERATORS:
            minimum += 1  # A crossing compares the prior and current observations.
        return minimum, used

    if operator == "not":
        if len(operands) != 1:
            raise ValueError("not requires exactly one rule operand")
    elif len(operands) < 2:
        raise ValueError(f"{operator} requires at least two rule operands")
    minimum, used = 1, set()
    for child in operands:
        child_minimum, child_used = _walk_rule(child, available_features)
        minimum = max(minimum, child_minimum)
        used.update(child_used)
    return minimum, used


def validate_hypothesis(hypothesis: StrategyHypothesis) -> ValidationReport:
    """Validate structure, causal references, supported inputs, and warm-up.

    Raises ``ValueError`` for unsupported or unsafe proposals. No market data,
    returns, labels, experiment results, or test-period metrics are accessed.
    """
    if not isinstance(hypothesis, StrategyHypothesis):
        raise TypeError("hypothesis must be a StrategyHypothesis")
    if hypothesis.schema_version != 1:
        raise ValueError("unsupported hypothesis schema version")
    if not isinstance(hypothesis.hypothesis_id, str) or not STABLE_ID.fullmatch(hypothesis.hypothesis_id):
        raise ValueError("hypothesis_id must be lowercase snake_case")
    if not isinstance(hypothesis.version, str) or not hypothesis.version.strip():
        raise ValueError("version must be nonempty")
    if hypothesis.strategy_family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported strategy family {hypothesis.strategy_family!r}")
    if not isinstance(hypothesis.rationale, str) or not hypothesis.rationale.strip():
        raise ValueError("rationale must be nonempty")
    if hypothesis.decision_timing != "after_signal_candle_close":
        raise ValueError("decision_timing must be 'after_signal_candle_close'")
    if not isinstance(hypothesis.exit_rule, ExitRule) or hypothesis.exit_rule.kind != "fixed_duration":
        raise ValueError("only fixed_duration exits are supported by the current backtester")
    holding = _positive_duration(hypothesis.exit_rule.max_holding_period, "max_holding_period")
    execution = hypothesis.execution_assumptions
    _validate_execution(execution)
    if holding < pd.Timedelta(execution.bar_duration):
        raise ValueError("max_holding_period must be at least one bar_duration")
    if isinstance(hypothesis.required_warmup_bars, bool) or not isinstance(hypothesis.required_warmup_bars, int):
        raise ValueError("required_warmup_bars must be an integer")
    if hypothesis.required_warmup_bars < 1:
        raise ValueError("required_warmup_bars must be positive")
    if any(not isinstance(item, str) or not item.strip()
           for item in (*hypothesis.assumptions, *hypothesis.known_limitations)):
        raise ValueError("assumptions and known_limitations must contain nonempty strings")

    output_to_warmup: dict[str, int] = {}
    indicator_warmups: dict[str, int] = {}
    for indicator in hypothesis.indicators:
        output, minimum = _indicator_output(indicator)
        if output in output_to_warmup:
            raise ValueError(f"duplicate indicator input for feature {output!r}")
        output_to_warmup[output] = minimum
        indicator_warmups[output] = minimum
    if not output_to_warmup:
        raise ValueError("at least one supported indicator input is required")

    available = set(output_to_warmup)
    all_used: set[str] = set()
    for rule in (hypothesis.long_entry, hypothesis.short_entry):
        _, used = _walk_rule(rule, available)
        all_used.update(used)
    indicator_names_used = all_used - SUPPORTED_PRICE_FIELDS
    unused = available - indicator_names_used
    if unused:
        raise ValueError(f"declared indicators are unused by entry rules: {sorted(unused)}")

    required_minimum = max(
        max(output_to_warmup.values()),
        *(_required_rule_bars(rule, output_to_warmup) for rule in
          (hypothesis.long_entry, hypothesis.short_entry)),
    )
    if hypothesis.required_warmup_bars < required_minimum:
        raise ValueError(
            f"required_warmup_bars={hypothesis.required_warmup_bars} is below minimum "
            f"{required_minimum} for selected indicators and rule offsets"
        )

    return ValidationReport(
        minimum_warmup_bars=required_minimum,
        indicator_warmup_bars=indicator_warmups,
        semantic_digest=hypothesis.semantic_digest(),
    )


def _feature_refs(rule: RuleNode) -> tuple[FeatureRef, ...]:
    refs: list[FeatureRef] = []
    for item in rule.operands:
        if isinstance(item, FeatureRef):
            refs.append(item)
        elif isinstance(item, RuleNode):
            refs.extend(_feature_refs(item))
    return tuple(refs)


def _required_rule_bars(rule: RuleNode, indicator_warmups: dict[str, int]) -> int:
    """Bars needed for this rule, including explicit lags and cross history."""
    operator = rule.operator
    if operator in COMPARISON_OPERATORS | CROSSING_OPERATORS:
        ref_requirements = [
            indicator_warmups.get(operand.name, 1) + operand.offset_bars
            for operand in rule.operands if isinstance(operand, FeatureRef)
        ]
        if operator in CROSSING_OPERATORS:
            if not ref_requirements:
                raise ValueError(f"{operator} requires at least one feature reference")
            ref_requirements = [bars + 1 for bars in ref_requirements]
        return max([1, *ref_requirements])
    return max(_required_rule_bars(child, indicator_warmups) for child in rule.operands)


def validate_unique_hypotheses(hypotheses: tuple[StrategyHypothesis, ...] | list[StrategyHypothesis]) -> None:
    """Reject behaviorally duplicate hypotheses after structural validation."""
    seen: dict[str, str] = {}
    for hypothesis in hypotheses:
        validate_hypothesis(hypothesis)
        digest = hypothesis.semantic_digest()
        if digest in seen:
            raise ValueError(
                f"duplicate hypothesis behavior: {hypothesis.hypothesis_id!r} duplicates {seen[digest]!r}"
            )
        seen[digest] = hypothesis.hypothesis_id
