"""Finite deterministic rule templates for the Stage 2 hypothesis engine.

This module generates declarative proposals only. It has no market-data,
experiment-store, backtester, or performance-metric dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import islice
import math
from typing import Iterator

from .hypothesis_schema import (
    ExecutionAssumptions,
    ExitRule,
    FeatureRef,
    IndicatorInput,
    RuleNode,
    StrategyHypothesis,
)
from .hypothesis_validator import (
    validate_hypothesis,
    validate_unique_hypotheses,
)


FAMILY_ORDER = (
    "trend_following", "mean_reversion", "breakout", "momentum", "volatility_filter",
)
MAX_CANDIDATE_LIMIT = 100
DEFAULT_CANDIDATE_LIMIT = 50

ALLOWED_EMA_PAIRS = ((9, 21), (9, 50), (21, 50))
ALLOWED_RSI_THRESHOLD_PAIRS = ((25.0, 75.0), (30.0, 70.0), (35.0, 65.0))
ALLOWED_MOMENTUM_PERIODS = (5, 15, 60)
ALLOWED_VOLATILITY_PERIODS = (15, 60)
ALLOWED_VOLATILITY_THRESHOLDS = (0.0002, 0.0005)
ALLOWED_VOLATILITY_MODES = ("below", "above")


@dataclass(frozen=True)
class TemplateConfig:
    """Finite, explicit choices for deterministic proposal enumeration."""

    families: tuple[str, ...] = FAMILY_ORDER
    ema_pairs: tuple[tuple[int, int], ...] = ALLOWED_EMA_PAIRS
    rsi_threshold_pairs: tuple[tuple[float, float], ...] = ALLOWED_RSI_THRESHOLD_PAIRS
    momentum_periods: tuple[int, ...] = ALLOWED_MOMENTUM_PERIODS
    volatility_periods: tuple[int, ...] = (15,)
    volatility_thresholds: tuple[float, ...] = ALLOWED_VOLATILITY_THRESHOLDS
    volatility_modes: tuple[str, ...] = ALLOWED_VOLATILITY_MODES
    volatility_return_periods: tuple[int, ...] = (5,)
    include_trend_volatility_filters: bool = False
    trend_volatility_periods: tuple[int, ...] = (15,)
    trend_volatility_thresholds: tuple[float, ...] = (0.0005,)
    trend_volatility_modes: tuple[str, ...] = ("below",)
    include_mean_reversion_volatility_filters: bool = False
    mean_reversion_volatility_periods: tuple[int, ...] = (15,)
    mean_reversion_volatility_thresholds: tuple[float, ...] = (0.0005,)
    mean_reversion_volatility_modes: tuple[str, ...] = ("below",)
    holding_period: str = "15min"
    execution_assumptions: ExecutionAssumptions = ExecutionAssumptions()
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT

    def __post_init__(self) -> None:
        for name in (
            "families", "ema_pairs", "rsi_threshold_pairs", "momentum_periods",
            "volatility_periods", "volatility_thresholds", "volatility_modes",
            "volatility_return_periods",
            "trend_volatility_periods", "trend_volatility_thresholds",
            "trend_volatility_modes",
            "mean_reversion_volatility_periods", "mean_reversion_volatility_thresholds",
            "mean_reversion_volatility_modes",
        ):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{name} must be a finite sequence")
            object.__setattr__(self, name, tuple(tuple(x) if name in {
                "ema_pairs", "rsi_threshold_pairs",
            } else x for x in value))
        if isinstance(self.candidate_limit, bool) or not isinstance(self.candidate_limit, int):
            raise ValueError("candidate_limit must be an integer")
        if not 1 <= self.candidate_limit <= MAX_CANDIDATE_LIMIT:
            raise ValueError(f"candidate_limit must be between 1 and {MAX_CANDIDATE_LIMIT}")
        if not self.families:
            raise ValueError("at least one family must be selected")
        if len(self.families) != len(set(self.families)):
            raise ValueError("families cannot contain duplicates")
        unknown = set(self.families) - set(FAMILY_ORDER)
        if unknown:
            raise ValueError(f"unsupported template families: {sorted(unknown)}")
        _bounded_choices("ema_pairs", self.ema_pairs, 10)
        _bounded_choices("rsi_threshold_pairs", self.rsi_threshold_pairs, 10)
        _bounded_choices("momentum_periods", self.momentum_periods, 10)
        _bounded_choices("volatility_periods", self.volatility_periods, 10)
        _bounded_choices("volatility_thresholds", self.volatility_thresholds, 10)
        _bounded_choices("volatility_modes", self.volatility_modes, 10)
        _bounded_choices("volatility_return_periods", self.volatility_return_periods, 10)
        if not isinstance(self.holding_period, str):
            raise ValueError("holding_period must be a duration string")
        if not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise ValueError("execution_assumptions must use ExecutionAssumptions")
        if not isinstance(self.include_trend_volatility_filters, bool):
            raise ValueError("include_trend_volatility_filters must be boolean")
        if not isinstance(self.include_mean_reversion_volatility_filters, bool):
            raise ValueError("include_mean_reversion_volatility_filters must be boolean")
        _validate_template_choices(self)


@dataclass(frozen=True)
class RejectedProposal:
    proposal_id: str
    reason: str


@dataclass(frozen=True)
class GenerationResult:
    hypotheses: tuple[StrategyHypothesis, ...]
    proposals_generated: int
    proposals_validated: int
    proposals_rejected: int
    proposals_deduplicated: int
    candidate_space_size: int
    not_generated_due_to_limit: int
    rejections: tuple[RejectedProposal, ...]


def _bounded_choices(name: str, values: tuple, maximum_size: int) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one choice")
    if len(values) > maximum_size:
        raise ValueError(f"{name} may contain at most {maximum_size} choices")


def _validate_template_choices(config: TemplateConfig) -> None:
    if not set(config.ema_pairs).issubset(ALLOWED_EMA_PAIRS):
        raise ValueError(f"ema_pairs choices must come from {ALLOWED_EMA_PAIRS}")
    if not set(config.rsi_threshold_pairs).issubset(ALLOWED_RSI_THRESHOLD_PAIRS):
        raise ValueError("rsi_threshold_pairs must use the predeclared bounded choices")
    if not set(config.momentum_periods).issubset(ALLOWED_MOMENTUM_PERIODS):
        raise ValueError(f"momentum_periods choices must come from {ALLOWED_MOMENTUM_PERIODS}")
    if not set(config.volatility_periods).issubset(ALLOWED_VOLATILITY_PERIODS):
        raise ValueError(f"volatility_periods choices must come from {ALLOWED_VOLATILITY_PERIODS}")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value <= 0
           for value in config.volatility_thresholds):
        raise ValueError("volatility thresholds must be finite positive values")
    if not set(config.volatility_thresholds).issubset(ALLOWED_VOLATILITY_THRESHOLDS):
        raise ValueError("volatility_thresholds must use the predeclared bounded choices")
    if not set(config.volatility_modes).issubset(ALLOWED_VOLATILITY_MODES):
        raise ValueError(f"volatility_modes choices must come from {ALLOWED_VOLATILITY_MODES}")
    if not set(config.volatility_return_periods).issubset(ALLOWED_MOMENTUM_PERIODS):
        raise ValueError("volatility_return_periods must use supported return periods")
    _bounded_choices("trend_volatility_periods", config.trend_volatility_periods, 10)
    _bounded_choices("trend_volatility_thresholds", config.trend_volatility_thresholds, 10)
    _bounded_choices("trend_volatility_modes", config.trend_volatility_modes, 10)
    if not set(config.trend_volatility_periods).issubset(ALLOWED_VOLATILITY_PERIODS):
        raise ValueError("trend_volatility_periods must use supported volatility periods")
    if not set(config.trend_volatility_thresholds).issubset(ALLOWED_VOLATILITY_THRESHOLDS):
        raise ValueError("trend_volatility_thresholds must use predeclared volatility thresholds")
    if not set(config.trend_volatility_modes).issubset(ALLOWED_VOLATILITY_MODES):
        raise ValueError("trend_volatility_modes must use supported volatility modes")
    _bounded_choices("mean_reversion_volatility_periods", config.mean_reversion_volatility_periods, 10)
    _bounded_choices("mean_reversion_volatility_thresholds", config.mean_reversion_volatility_thresholds, 10)
    _bounded_choices("mean_reversion_volatility_modes", config.mean_reversion_volatility_modes, 10)
    if not set(config.mean_reversion_volatility_periods).issubset(ALLOWED_VOLATILITY_PERIODS):
        raise ValueError("mean_reversion_volatility_periods must use supported volatility periods")
    if not set(config.mean_reversion_volatility_thresholds).issubset(ALLOWED_VOLATILITY_THRESHOLDS):
        raise ValueError("mean_reversion_volatility_thresholds must use predeclared volatility thresholds")
    if not set(config.mean_reversion_volatility_modes).issubset(ALLOWED_VOLATILITY_MODES):
        raise ValueError("mean_reversion_volatility_modes must use supported volatility modes")
    try:
        import pandas as pd

        hold = pd.Timedelta(config.holding_period)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("holding_period must be a valid positive duration") from exc
    if pd.isna(hold) or hold <= pd.Timedelta(0):
        raise ValueError("holding_period must be a valid positive duration")


def _families(config: TemplateConfig) -> tuple[str, ...]:
    selected = set(config.families)
    return tuple(family for family in FAMILY_ORDER if family in selected)


def _count_family(config: TemplateConfig, family: str) -> int:
    if family == "trend_following":
        return len(config.ema_pairs) + (len(config.ema_pairs) *
            len(config.trend_volatility_periods) * len(config.trend_volatility_thresholds)
            * len(config.trend_volatility_modes) if config.include_trend_volatility_filters else 0)
    if family == "mean_reversion":
        return len(config.rsi_threshold_pairs) + (
            len(config.rsi_threshold_pairs) * len(config.mean_reversion_volatility_periods)
            * len(config.mean_reversion_volatility_thresholds)
            * len(config.mean_reversion_volatility_modes)
            if config.include_mean_reversion_volatility_filters else 0
        )
    return {
        "breakout": 1,
        "momentum": len(config.momentum_periods),
        "volatility_filter": (
            len(config.volatility_periods) * len(config.volatility_thresholds)
            * len(config.volatility_modes) * len(config.volatility_return_periods)
        ),
    }[family]


def candidate_space_size(config: TemplateConfig) -> int:
    """Number of raw proposals in this finite configuration, before the limit."""
    return sum(_count_family(config, family) for family in _families(config))


def _rule(operator: str, left, right) -> RuleNode:
    return RuleNode(operator, (left, right))


def _boolean(operator: str, *rules: RuleNode) -> RuleNode:
    return RuleNode(operator, tuple(rules))


def _make_hypothesis(
    config: TemplateConfig,
    family: str,
    indicators: tuple[IndicatorInput, ...],
    long_entry: RuleNode,
    short_entry: RuleNode,
    warmup: int,
    rationale: str,
    assumptions: tuple[str, ...] = (),
) -> StrategyHypothesis:
    hypothesis = StrategyHypothesis(
        hypothesis_id="pending_hypothesis_id",
        version="1.0.0",
        strategy_family=family,
        rationale=rationale,
        indicators=indicators,
        long_entry=long_entry,
        short_entry=short_entry,
        decision_timing="after_signal_candle_close",
        exit_rule=ExitRule("fixed_duration", config.holding_period),
        required_warmup_bars=warmup,
        execution_assumptions=config.execution_assumptions,
        assumptions=assumptions,
        known_limitations=(
            "Declarative proposal only; it has not been compiled, backtested, or evaluated.",
            "Fixed transaction cost and OHLC fill assumptions are not verified execution costs.",
        ),
    )
    digest = hypothesis.semantic_digest()
    # ID is display/tracking metadata and is excluded from semantic behavior.
    identified = replace(hypothesis, hypothesis_id=f"{family}_{digest[:16]}")
    return identified


def _iter_proposals(config: TemplateConfig) -> Iterator[StrategyHypothesis]:
    for family in _families(config):
        if family == "trend_following":
            for fast, slow in config.ema_pairs:
                inputs = (IndicatorInput("ema", {"span": fast}), IndicatorInput("ema", {"span": slow}))
                long_rule = _rule("crosses_above", FeatureRef(f"ema_{fast}"), FeatureRef(f"ema_{slow}"))
                short_rule = _rule("crosses_below", FeatureRef(f"ema_{fast}"), FeatureRef(f"ema_{slow}"))
                yield _make_hypothesis(
                    config, family, inputs, long_rule, short_rule, 5 * slow + 1,
                    f"A fast EMA {fast} crossing the slower EMA {slow} may indicate directional persistence.",
                )
            if config.include_trend_volatility_filters:
                for fast, slow in config.ema_pairs:
                    for period in config.trend_volatility_periods:
                        for threshold in config.trend_volatility_thresholds:
                            for mode in config.trend_volatility_modes:
                                comparison = "lt" if mode == "below" else "gt"
                                regime = _rule(comparison, FeatureRef(f"volatility_{period}"), threshold)
                                long_rule = _boolean("and", _rule(
                                    "crosses_above", FeatureRef(f"ema_{fast}"), FeatureRef(f"ema_{slow}"),
                                ), regime)
                                short_rule = _boolean("and", _rule(
                                    "crosses_below", FeatureRef(f"ema_{fast}"), FeatureRef(f"ema_{slow}"),
                                ), regime)
                                inputs = (
                                    IndicatorInput("ema", {"span": fast}),
                                    IndicatorInput("ema", {"span": slow}),
                                    IndicatorInput("volatility", {"period": period}),
                                )
                                yield _make_hypothesis(
                                    config, family, inputs, long_rule, short_rule,
                                    max(5 * slow + 1, period + 1),
                                    f"EMA {fast}/{slow} crossover conditioned on {period}-bar volatility {mode} {threshold:g}.",
                                    assumptions=("Volatility conditioning is a predeclared hypothesis component, not a selected regime.",),
                                )
        elif family == "mean_reversion":
            for long_level, short_level in config.rsi_threshold_pairs:
                indicator = IndicatorInput("rsi", {"period": 14})
                long_rule = _rule("crosses_above", FeatureRef("rsi_14"), long_level)
                short_rule = _rule("crosses_below", FeatureRef("rsi_14"), short_level)
                yield _make_hypothesis(
                    config, family, (indicator,), long_rule, short_rule, 16,
                    f"RSI(14) recrossing {long_level:g}/{short_level:g} may mark a short-horizon reversal.",
                )
            if config.include_mean_reversion_volatility_filters:
                for long_level, short_level in config.rsi_threshold_pairs:
                    for period in config.mean_reversion_volatility_periods:
                        for threshold in config.mean_reversion_volatility_thresholds:
                            for mode in config.mean_reversion_volatility_modes:
                                comparison = "lt" if mode == "below" else "gt"
                                regime = _rule(comparison, FeatureRef(f"volatility_{period}"), threshold)
                                long_rule = _boolean("and", _rule(
                                    "crosses_above", FeatureRef("rsi_14"), long_level,
                                ), regime)
                                short_rule = _boolean("and", _rule(
                                    "crosses_below", FeatureRef("rsi_14"), short_level,
                                ), regime)
                                yield _make_hypothesis(
                                    config, family,
                                    (IndicatorInput("rsi", {"period": 14}),
                                     IndicatorInput("volatility", {"period": period})),
                                    long_rule, short_rule, max(16, period + 1),
                                    f"RSI(14) mean reversion conditioned on {period}-bar volatility {mode} {threshold:g}.",
                                    assumptions=("Low-volatility conditioning is a predeclared response to training-only cost drag; it is not a selected regime.",),
                                )
        elif family == "breakout":
            # This is explicitly a one-prior-candle break, not Donchian(20).
            indicator = IndicatorInput("return", {"period": 1})
            long_rule = _boolean(
                "and", _rule("gt", FeatureRef("close"), FeatureRef("high", 1)),
                _rule("gt", FeatureRef("return_1"), 0.0),
            )
            short_rule = _boolean(
                "and", _rule("lt", FeatureRef("close"), FeatureRef("low", 1)),
                _rule("lt", FeatureRef("return_1"), 0.0),
            )
            yield _make_hypothesis(
                config, family, (indicator,), long_rule, short_rule, 2,
                "A completed close beyond the previous candle extreme with matching one-bar return may indicate breakout continuation.",
            )
        elif family == "momentum":
            for period in config.momentum_periods:
                indicator = IndicatorInput("return", {"period": period})
                feature = FeatureRef(f"return_{period}")
                long_rule = _rule("gt", feature, 0.0)
                short_rule = _rule("lt", feature, 0.0)
                yield _make_hypothesis(
                    config, family, (indicator,), long_rule, short_rule, period + 1,
                    f"A positive or negative {period}-bar return may persist over the fixed holding interval.",
                )
        elif family == "volatility_filter":
            for vol_period in config.volatility_periods:
                for threshold in config.volatility_thresholds:
                    for mode in config.volatility_modes:
                        for return_period in config.volatility_return_periods:
                            return_feature = FeatureRef(f"return_{return_period}")
                            volatility_feature = FeatureRef(f"volatility_{vol_period}")
                            comparison = "lt" if mode == "below" else "gt"
                            indicators = (
                                IndicatorInput("return", {"period": return_period}),
                                IndicatorInput("volatility", {"period": vol_period}),
                            )
                            long_rule = _boolean(
                                "and", _rule("gt", return_feature, 0.0),
                                _rule(comparison, volatility_feature, threshold),
                            )
                            short_rule = _boolean(
                                "and", _rule("lt", return_feature, 0.0),
                                _rule(comparison, volatility_feature, threshold),
                            )
                            warmup = max(return_period + 1, vol_period + 1)
                            yield _make_hypothesis(
                                config, family, indicators, long_rule, short_rule, warmup,
                                f"A {return_period}-bar directional return conditioned on {vol_period}-bar volatility {mode} {threshold:g} may distinguish regimes.",
                            )


def generate_hypotheses(config: TemplateConfig = TemplateConfig()) -> GenerationResult:
    """Generate, validate, and semantically deduplicate a bounded proposal set.

    The candidate limit applies to raw proposals examined. Invalid proposals
    count as generated and rejected; valid duplicate behaviors count as
    validated and deduplicated. The first valid proposal for each digest is kept.
    """
    if not isinstance(config, TemplateConfig):
        raise TypeError("config must be a TemplateConfig")
    total_space = candidate_space_size(config)
    accepted: list[StrategyHypothesis] = []
    rejection_rows: list[RejectedProposal] = []
    seen: dict[str, str] = {}
    generated = validated = rejected = deduplicated = 0

    for proposal in islice(_iter_proposals(config), config.candidate_limit):
        generated += 1
        try:
            validation = validate_hypothesis(proposal)
        except (TypeError, ValueError) as exc:
            rejected += 1
            rejection_rows.append(RejectedProposal(proposal.hypothesis_id, str(exc)))
            continue
        validated += 1
        digest = validation.semantic_digest
        if digest in seen:
            deduplicated += 1
            continue
        seen[digest] = proposal.hypothesis_id
        accepted.append(proposal)

    # This is a defensive cross-check on retained outputs, not a second filter.
    validate_unique_hypotheses(accepted)
    return GenerationResult(
        hypotheses=tuple(accepted),
        proposals_generated=generated,
        proposals_validated=validated,
        proposals_rejected=rejected,
        proposals_deduplicated=deduplicated,
        candidate_space_size=total_space,
        not_generated_due_to_limit=max(0, total_space - generated),
        rejections=tuple(rejection_rows),
    )
