"""Immutable, JSON-serializable schema for declarative strategy hypotheses.

Rules are data trees only. This module deliberately provides no expression
evaluator, Python-code loader, or dynamic execution facility.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from typing import Any, TypeAlias


SCHEMA_VERSION = 1


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, init=False)
class FrozenMapping(Mapping[str, Any]):
    """Small recursively immutable mapping used for JSON parameter objects."""

    _items: tuple[tuple[str, Any], ...]

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        if value is None:
            value = {}
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise TypeError("mapping keys must be strings")
        items = tuple(sorted((key, _freeze(item)) for key, item in value.items()))
        object.__setattr__(self, "_items", items)

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class FeatureRef:
    """Reference an available feature at this bar (offset 0) or in its past."""

    name: str
    offset_bars: int = 0


RuleValue: TypeAlias = Any


@dataclass(frozen=True)
class RuleNode:
    """One operator and its immutable operands; validation is structural."""

    operator: str
    operands: tuple[RuleValue, ...]

    def __post_init__(self) -> None:
        if isinstance(self.operands, (str, bytes)):
            raise TypeError("rule operands must be a sequence, not text")
        operands = tuple(self.operands)
        if any(not isinstance(item, (RuleNode, FeatureRef, str, bool, int, float))
               and item is not None for item in operands):
            raise TypeError("rule operands must be rule nodes, feature references, or JSON scalars")
        object.__setattr__(self, "operands", operands)


@dataclass(frozen=True, init=False)
class IndicatorInput:
    """A named pipeline indicator and its immutable configuration."""

    name: str
    parameters: FrozenMapping

    def __init__(self, name: str, parameters: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parameters", FrozenMapping(parameters))


@dataclass(frozen=True)
class ExitRule:
    """The v1 backtester-supported fixed-duration exit contract."""

    kind: str
    max_holding_period: str


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Execution fields that map directly to the current OHLC backtester."""

    timestamp_convention: str = "bar_open"
    bar_duration: str = "1min"
    execution_delay: str = "0s"
    allow_overlapping: bool = False
    transaction_cost_pips: float = 1.0
    pip_size: float = 0.0001
    gap_tolerance: str = "90s"


@dataclass(frozen=True)
class StrategyHypothesis:
    """Versioned declarative proposal; it is not an executable strategy."""

    hypothesis_id: str
    version: str
    strategy_family: str
    rationale: str
    indicators: tuple[IndicatorInput, ...]
    long_entry: RuleNode
    short_entry: RuleNode
    decision_timing: str
    exit_rule: ExitRule
    required_warmup_bars: int
    execution_assumptions: ExecutionAssumptions
    assumptions: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.indicators, (str, bytes)):
            raise TypeError("indicators must be a sequence of IndicatorInput values")
        if isinstance(self.assumptions, (str, bytes)) or isinstance(self.known_limitations, (str, bytes)):
            raise TypeError("assumptions and known_limitations must be sequences of strings")
        object.__setattr__(self, "indicators", tuple(self.indicators))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "known_limitations", tuple(self.known_limitations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "version": self.version,
            "strategy_family": self.strategy_family,
            "rationale": self.rationale,
            "indicators": [
                {"name": indicator.name, "parameters": _thaw(indicator.parameters)}
                for indicator in self.indicators
            ],
            "long_entry": rule_to_dict(self.long_entry),
            "short_entry": rule_to_dict(self.short_entry),
            "decision_timing": self.decision_timing,
            "exit_rule": {
                "kind": self.exit_rule.kind,
                "max_holding_period": self.exit_rule.max_holding_period,
            },
            "required_warmup_bars": self.required_warmup_bars,
            "execution_assumptions": {
                "timestamp_convention": self.execution_assumptions.timestamp_convention,
                "bar_duration": self.execution_assumptions.bar_duration,
                "execution_delay": self.execution_assumptions.execution_delay,
                "allow_overlapping": self.execution_assumptions.allow_overlapping,
                "transaction_cost_pips": self.execution_assumptions.transaction_cost_pips,
                "pip_size": self.execution_assumptions.pip_size,
                "gap_tolerance": self.execution_assumptions.gap_tolerance,
            },
            "assumptions": list(self.assumptions),
            "known_limitations": list(self.known_limitations),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyHypothesis":
        expected = {
            "schema_version", "hypothesis_id", "version", "strategy_family", "rationale",
            "indicators", "long_entry", "short_entry", "decision_timing", "exit_rule",
            "required_warmup_bars", "execution_assumptions", "assumptions", "known_limitations",
        }
        _require_exact_keys(value, expected, "hypothesis")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported hypothesis schema_version {value['schema_version']!r}")
        indicators = []
        if not isinstance(value["indicators"], (list, tuple)):
            raise ValueError("indicators must be a list")
        for item in value["indicators"]:
            _require_exact_keys(item, {"name", "parameters"}, "indicator")
            indicators.append(IndicatorInput(item["name"], item["parameters"]))
        exit_value = value["exit_rule"]
        _require_exact_keys(exit_value, {"kind", "max_holding_period"}, "exit_rule")
        execution = value["execution_assumptions"]
        _require_exact_keys(execution, {
            "timestamp_convention", "bar_duration", "execution_delay", "allow_overlapping",
            "transaction_cost_pips", "pip_size", "gap_tolerance",
        }, "execution_assumptions")
        return cls(
            hypothesis_id=value["hypothesis_id"], version=value["version"],
            strategy_family=value["strategy_family"], rationale=value["rationale"],
            indicators=tuple(indicators), long_entry=rule_from_dict(value["long_entry"]),
            short_entry=rule_from_dict(value["short_entry"]),
            decision_timing=value["decision_timing"],
            exit_rule=ExitRule(exit_value["kind"], exit_value["max_holding_period"]),
            required_warmup_bars=value["required_warmup_bars"],
            execution_assumptions=ExecutionAssumptions(**execution),
            assumptions=tuple(value["assumptions"]),
            known_limitations=tuple(value["known_limitations"]),
            schema_version=value["schema_version"],
        )

    def canonical_behavior(self) -> dict[str, Any]:
        """Return normalized executable behavior, excluding identity/display text."""
        indicators = [
            {"name": indicator.name, "parameters": _thaw(indicator.parameters)}
            for indicator in self.indicators
        ]
        indicators.sort(key=lambda item: (item["name"], canonical_json(item["parameters"])))
        return {
            "schema_version": self.schema_version,
            "indicators": indicators,
            "long_entry": canonical_rule(self.long_entry),
            "short_entry": canonical_rule(self.short_entry),
            "decision_timing": self.decision_timing,
            "exit_rule": {
                "kind": self.exit_rule.kind,
                "max_holding_period": _canonical_duration(self.exit_rule.max_holding_period),
            },
            "required_warmup_bars": self.required_warmup_bars,
            "execution_assumptions": {
                "timestamp_convention": self.execution_assumptions.timestamp_convention,
                "bar_duration": _canonical_duration(self.execution_assumptions.bar_duration),
                "execution_delay": _canonical_duration(self.execution_assumptions.execution_delay),
                "allow_overlapping": self.execution_assumptions.allow_overlapping,
                "transaction_cost_pips": canonical_number(self.execution_assumptions.transaction_cost_pips),
                "pip_size": canonical_number(self.execution_assumptions.pip_size),
                "gap_tolerance": _canonical_duration(self.execution_assumptions.gap_tolerance),
            },
        }

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_behavior())

    def semantic_digest(self) -> str:
        import hashlib

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def rule_to_dict(value: RuleValue) -> Any:
    if isinstance(value, FeatureRef):
        return {"feature": value.name, "offset_bars": value.offset_bars}
    if isinstance(value, RuleNode):
        return {"operator": value.operator, "operands": [rule_to_dict(item) for item in value.operands]}
    return value


def rule_from_dict(value: Any) -> RuleValue:
    if isinstance(value, Mapping):
        if set(value) == {"feature", "offset_bars"}:
            return FeatureRef(value["feature"], value["offset_bars"])
        if set(value) == {"operator", "operands"}:
            if not isinstance(value["operands"], (list, tuple)):
                raise ValueError("rule operands must be a list")
            return RuleNode(value["operator"], tuple(rule_from_dict(item) for item in value["operands"]))
        raise ValueError("rule object must be a feature reference or operator node")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("rule values must be feature references, operator nodes, or JSON scalars")


def canonical_rule(value: RuleValue) -> Any:
    """Canonicalize rule trees, including commutative Boolean child order."""
    if isinstance(value, FeatureRef):
        return {"feature": value.name, "offset_bars": value.offset_bars}
    if isinstance(value, RuleNode):
        children = [canonical_rule(child) for child in value.operands]
        if value.operator in {"and", "or"}:
            flattened = []
            for child in children:
                if isinstance(child, dict) and child.get("operator") == value.operator:
                    flattened.extend(child["operands"])
                else:
                    flattened.append(child)
            children = sorted(flattened, key=canonical_json)
        return {"operator": value.operator, "operands": children}
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    return value


def canonical_number(value: int | float) -> int | float:
    """Normalize numerically equivalent integer and integral-float inputs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_json(value: Any) -> str:
    """Stable compact JSON for already-validated JSON-compatible values."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical behavior must contain finite JSON-compatible values") from exc


def _canonical_duration(value: str) -> str:
    # Kept local to avoid adding a schema module dependency beyond the project stack.
    import pandas as pd

    try:
        parsed = pd.Timedelta(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid duration {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"invalid duration {value!r}")
    return parsed.isoformat()
