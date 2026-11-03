"""Small, validated registry of the project's predeclared strategy families."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


_STABLE_ID = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    family: str
    hypothesis: str
    rule_description: str
    parameters: Mapping[str, Any]
    version: str

    def __post_init__(self) -> None:
        if not _STABLE_ID.fullmatch(self.strategy_id):
            raise ValueError("strategy_id must be a stable lowercase snake_case identifier")
        if not _STABLE_ID.fullmatch(self.family):
            raise ValueError("family must be a lowercase snake_case identifier")
        for field_name in ("name", "hypothesis", "rule_description", "version"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be a nonempty string")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        try:
            json.dumps(dict(self.parameters), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must contain finite JSON-compatible values") from exc


_SPECS = (
    StrategySpec(
        strategy_id="ema_crossover",
        name="EMA 9/21 crossover",
        family="trend_following",
        hypothesis="Short-term directional momentum may persist after a fast/slow EMA crossover.",
        rule_description=(
            "After a candle closes, long when EMA 9 crosses from at/below EMA 21 to above; "
            "short for the reverse cross. Enter at the first eligible candle open strictly "
            "after the decision timestamp. Hold for 15 minutes."
        ),
        parameters={"fast_ema": 9, "slow_ema": 21, "holding_minutes": 15, "warmup_bars": 105},
        version="1.0.0",
    ),
    StrategySpec(
        strategy_id="rsi_mean_reversion",
        name="RSI(14) mean reversion",
        family="mean_reversion",
        hypothesis="A recross out of an RSI extreme may mark a short-horizon reversal.",
        rule_description=(
            "After a candle closes, long when RSI(14) crosses back above 30; short when it "
            "crosses back below 70. Enter at the first eligible candle open strictly after "
            "the decision timestamp. Hold for 15 minutes."
        ),
        parameters={"rsi_period": 14, "long_recross": 30.0, "short_recross": 70.0,
                    "holding_minutes": 15, "warmup_bars": 105},
        version="1.0.0",
    ),
    StrategySpec(
        strategy_id="donchian_breakout",
        name="Donchian 20 breakout",
        family="breakout",
        hypothesis="A close beyond the preceding price channel may indicate continuation.",
        rule_description=(
            "After a candle closes, long above the highest high of the prior 20 candles or "
            "short below the lowest low of the prior 20 candles; exclude the current candle "
            "from the channel. Enter at the first eligible candle open strictly after the "
            "decision timestamp. Hold for 15 minutes."
        ),
        parameters={"donchian_period": 20, "holding_minutes": 15, "warmup_bars": 105},
        version="1.0.0",
    ),
)

STRATEGIES: tuple[StrategySpec, ...] = _SPECS
_BY_ID = {spec.strategy_id: spec for spec in _SPECS}


def get_strategy(strategy_id: str) -> StrategySpec:
    """Return a registered strategy or raise a descriptive KeyError."""
    try:
        return _BY_ID[strategy_id]
    except KeyError as exc:
        raise KeyError(f"Unregistered strategy_id: {strategy_id!r}") from exc


def list_strategies() -> tuple[StrategySpec, ...]:
    return STRATEGIES


def registry_as_dicts() -> list[dict[str, Any]]:
    return [
        {"strategy_id": s.strategy_id, "name": s.name, "family": s.family,
         "hypothesis": s.hypothesis, "rule_description": s.rule_description,
         "parameters": dict(s.parameters), "version": s.version}
        for s in STRATEGIES
    ]


def validate_registry_configuration(
    runner_configurations: Mapping[str, Mapping[str, Any]],
    *,
    specs: Sequence[StrategySpec] = STRATEGIES,
) -> None:
    """Fail clearly if runner constants drift from registered parameters.

    ``runner_configurations`` maps stable strategy IDs to the exact parameter
    values used by their current signal/evaluation implementation.
    """
    by_id = {spec.strategy_id: spec for spec in specs}
    for strategy_id, actual in runner_configurations.items():
        if strategy_id not in by_id:
            raise ValueError(f"Runner configuration references unregistered strategy {strategy_id!r}")
        registered = dict(by_id[strategy_id].parameters)
        actual_values = dict(actual)
        if actual_values != registered:
            differing = sorted(
                key for key in set(actual_values) | set(registered)
                if actual_values.get(key, object()) != registered.get(key, object())
            )
            details = ", ".join(
                f"{key}: registry={registered.get(key)!r}, runner={actual_values.get(key)!r}"
                for key in differing
            )
            raise ValueError(f"Strategy configuration drift for {strategy_id}: {details}")
