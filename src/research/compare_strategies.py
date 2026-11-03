"""Compare three predeclared causal strategies under shared execution rules.

Run from the repository root with ``python -m src.research.compare_strategies``.
The candidate rules and parameters are fixed in this module. The test metrics
are labeled as previously accessed because the earlier EMA report included
that period; this script contains no strategy-selection logic.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import DEFAULT_DATA_PATH, load_market_data
from .experiment_store import (
    ExperimentStore,
    ExecutionClaimLost,
    PROJECT_ROOT as PROJECT_ROOT_FOR_STORE,
    RunHandle,
    code_provenance,
    dataset_metadata,
    utc_date_range,
)
from .experiment_ema import (
    BAR_DURATION,
    FAST_EMA,
    HOLDING_PERIOD,
    SLOW_EMA,
    TIMESTAMP_CONVENTION,
    WARMUP_BARS,
    evaluate_period,
    split_metadata,
    timeline_periods,
)
from .features import (
    build_features,
    indicator_segment_ids,
    segment_bar_counts,
)
from .strategy_registry import (
    get_strategy,
    list_strategies,
    validate_registry_configuration,
)


TRANSACTION_COST_PIPS = 1.0
EXECUTION_DELAY = "0s"
GAP_TOLERANCE = "90s"
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 30.0
RSI_SHORT_THRESHOLD = 70.0
DONCHIAN_PERIOD = 20
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "strategy_comparison.md"
STRATEGIES = tuple(spec.name for spec in list_strategies())
STRATEGY_IDS = {spec.name: spec.strategy_id for spec in list_strategies()}
SIGNAL_ID_PREFIXES = {
    STRATEGIES[0]: "ema_9_21", STRATEGIES[1]: "rsi_14", STRATEGIES[2]: "donchian_20",
}
RUNNER_CONFIGURATIONS = {
    "ema_crossover": {"fast_ema": FAST_EMA, "slow_ema": SLOW_EMA,
                      "holding_minutes": int(HOLDING_PERIOD.total_seconds() / 60),
                      "warmup_bars": WARMUP_BARS},
    "rsi_mean_reversion": {"rsi_period": RSI_PERIOD, "long_recross": RSI_LONG_THRESHOLD,
                            "short_recross": RSI_SHORT_THRESHOLD,
                            "holding_minutes": int(HOLDING_PERIOD.total_seconds() / 60),
                            "warmup_bars": WARMUP_BARS},
    "donchian_breakout": {"donchian_period": DONCHIAN_PERIOD,
                          "holding_minutes": int(HOLDING_PERIOD.total_seconds() / 60),
                          "warmup_bars": WARMUP_BARS},
}


def validate_selected_strategy_ids(strategy_ids: Sequence[str]) -> list[Any]:
    """Validate a bounded selection and its fixed runner/registry configuration."""
    selected_ids = list(strategy_ids)
    if not selected_ids:
        raise ValueError("at least one strategy_id is required")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("duplicate strategy IDs are not allowed")
    specs = [get_strategy(strategy_id) for strategy_id in selected_ids]
    validate_registry_configuration({key: RUNNER_CONFIGURATIONS[key] for key in selected_ids})
    return specs


@dataclass
class ComparisonRunResult:
    """Structured execution result for callers that coordinate multiple runs."""

    handles: dict[str, RunHandle]
    periods: dict[str, dict[str, Any]]
    evaluations: dict[str, dict[str, dict[str, Any]]]
    market: pd.DataFrame
    report_path: Path | None = None


def prepare_comparison_context(market: pd.DataFrame | None = None) -> dict[str, Any]:
    """Load/fingerprint shared inputs once; this does not generate signals."""
    data = load_market_data() if market is None else market
    fingerprint, identity = dataset_metadata(DEFAULT_DATA_PATH, data)
    date_start, date_end = utc_date_range(data)
    return {
        "market": data,
        "fingerprint": fingerprint,
        "identity": identity,
        "date_start": date_start,
        "date_end": date_end,
        "quality": dict(data.attrs.get("load_report", {})),
        "software": code_provenance(PROJECT_ROOT_FOR_STORE),
        "periods": timeline_periods(data),
    }


def comparison_request_args(strategy_id: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical comparison-run request used by preview and execution."""
    spec = get_strategy(strategy_id)
    return {
        "strategy": spec,
        "parameters": spec.parameters,
        "dataset_fingerprint": context["fingerprint"],
        "dataset_identity": context["identity"],
        "date_start_utc": context["date_start"],
        "date_end_utc": context["date_end"],
        "execution_assumptions": {
            "experiment_kind": "strategy_family_comparison",
            "timestamp_convention": TIMESTAMP_CONVENTION,
            "bar_duration": str(BAR_DURATION),
            "holding_period": str(HOLDING_PERIOD),
            "execution_delay": EXECUTION_DELAY,
            "allow_overlapping": False,
            "transaction_cost_pips_round_trip": TRANSACTION_COST_PIPS,
            "pip_size": 0.0001,
        },
        "gap_policy": {"tolerance": GAP_TOLERANCE, "indicator_reset_above": "90s",
                       "exact_boundary_resets": False},
        "software": context["software"],
        "data_quality": context["quality"],
        "split_metadata": split_metadata(context["periods"]),
    }


def generate_strategy_signals(
    market_data: pd.DataFrame,
    *,
    strategy_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build close-time candidate events and mark segment-local warm-up readiness.

    Events before 105 observed candles in their segment are not passed to the
    backtester. This exceeds the finite RSI(14), volatility(60), and Donchian(20)
    windows and is five EMA 21 spans. It is a practical recursive-EMA convention,
    not an exact finite point at which seed influence becomes zero. The generic
    EMA 50 feature is not five-span-warmed by this strategy-specific gate.
    """
    features = build_features(market_data)
    segments = indicator_segment_ids(features["time"])
    bars_in_segment = segment_bar_counts(segments)
    close = features["close"]

    ema_spread = features["ema_9"] - features["ema_21"]
    previous_ema_spread = ema_spread.shift(1)
    same_segment = segments.eq(segments.shift(1))
    ema_long = same_segment & (previous_ema_spread <= 0) & (ema_spread > 0)
    ema_short = same_segment & (previous_ema_spread >= 0) & (ema_spread < 0)

    rsi = features["rsi_14"]
    previous_rsi = rsi.shift(1)
    rsi_long = same_segment & (previous_rsi <= RSI_LONG_THRESHOLD) & (rsi > RSI_LONG_THRESHOLD)
    rsi_short = same_segment & (previous_rsi >= RSI_SHORT_THRESHOLD) & (rsi < RSI_SHORT_THRESHOLD)

    # Shift after rolling so the current candle cannot set its own breakout level.
    prior_high = features["high"].groupby(segments, sort=False).transform(
        lambda values: values.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().shift(1)
    )
    prior_low = features["low"].groupby(segments, sort=False).transform(
        lambda values: values.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().shift(1)
    )
    donchian_long = close > prior_high
    donchian_short = close < prior_low

    rules = {
        STRATEGIES[0]: (ema_long, ema_short),
        STRATEGIES[1]: (rsi_long, rsi_short),
        STRATEGIES[2]: (donchian_long, donchian_short),
    }
    signals: dict[str, pd.DataFrame] = {}
    requested = set(STRATEGIES if strategy_names is None else strategy_names)
    unknown = requested - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategy names: {sorted(unknown)}")
    for name, (long_rule, short_rule) in rules.items():
        if name not in requested:
            continue
        directions = np.select([long_rule, short_rule], [1, -1], default=0)
        selected = np.flatnonzero(directions != 0)
        source_times = features["time"].iloc[selected].reset_index(drop=True)
        slug = {
            STRATEGIES[0]: "ema_9_21",
            STRATEGIES[1]: "rsi_14",
            STRATEGIES[2]: "donchian_20",
        }[name]
        segment_counts = bars_in_segment.iloc[selected].to_numpy()
        signals[name] = pd.DataFrame({
            "signal_id": [f"{slug}:{timestamp.isoformat()}" for timestamp in source_times],
            "decision_timestamp": (
                source_times + BAR_DURATION
            ),
            "direction": np.where(directions[selected] > 0, "long", "short"),
            "source_row_index": selected,
            "source_index": selected,
            "source_candle_timestamp": source_times,
            "indicator_segment_id": segments.iloc[selected].to_numpy(),
            "segment_bar_count": segment_counts,
            "indicator_warmup_complete": segment_counts >= WARMUP_BARS,
        })
    return signals


def evaluate_all_periods(
    market: pd.DataFrame,
    strategy_signals: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Evaluate train and validation first, then the accessed test descriptively."""
    periods = timeline_periods(market)
    evaluations: dict[str, dict[str, Any]] = {}
    for period_name in ("train", "validation", "test"):
        evaluations[period_name] = {}
        for strategy_name in STRATEGIES:
            evaluations[period_name][strategy_name] = evaluate_period(
                market,
                strategy_signals[strategy_name],
                periods[period_name],
                execution_delay=EXECUTION_DELAY,
                gap_tolerance=GAP_TOLERANCE,
                transaction_cost_pips=TRANSACTION_COST_PIPS,
                expected_signal_id_prefix=SIGNAL_ID_PREFIXES[strategy_name],
            )
    return periods, evaluations


def _metric(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)) and np.isinf(value):
        return "infinity"
    return f"{float(value) * 100:.2f}%" if percent else f"{float(value):.4f}"


def _rejection_counts(evaluation: dict[str, Any]) -> str:
    reasons = evaluation["rejected"]
    return ", ".join((
        f"unavailable-entry={reasons['unavailable_entry']}",
        f"unavailable-exit={reasons['unavailable_exit']}",
        f"gap={reasons['gap_exceeds_tolerance']}",
        f"overlap={reasons['overlap_skipped']}",
        f"other={reasons['invalid_decision_timestamp'] + reasons['invalid_direction'] + reasons['invalid_entry_price'] + reasons['invalid_exit_price']}",
        f"backtester-total={sum(reasons.values())}",
    ))


def render_report(
    periods: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    *,
    market: pd.DataFrame,
    strategy_names: Sequence[str] | None = None,
) -> str:
    strategy_names = tuple(STRATEGIES if strategy_names is None else strategy_names)
    lines = [
        "# Controlled Strategy Family Comparison",
        "",
        "Results below were produced with segment-local indicator resets at gaps greater than 90 seconds.",
        "",
        "This report is descriptive. It does not establish statistical significance, a robust edge, or profitability.",
        "",
        "## Predeclared rules and shared configuration",
        "",
        "| Candidate | Fixed signal rule |",
        "|---|---|",
        "| EMA trend following | Long when EMA 9 crosses from at/below EMA 21 to above; short for the reverse cross. |",
        "| RSI mean reversion | RSI(14) crosses back above 30 for long; crosses back below 70 for short. |",
        "| Donchian breakout | Long when close exceeds the prior 20 candles' highest high; short when close falls below the prior 20 candles' lowest low. Current candle is excluded from the channel. |",
        "",
        f"- Timestamp convention: UTC-aware Dukascopy M1 `bar_open`; decision timestamp is the signal candle close ({BAR_DURATION} after its open).",
        f"- Entry: first eligible candle open strictly after decision time; execution delay {EXECUTION_DELAY}.",
        f"- Fixed holding period: {HOLDING_PERIOD}; gap tolerance: {GAP_TOLERANCE}; overlapping positions disabled.",
        f"- Fixed cost: {TRANSACTION_COST_PIPS:.1f} pips per round trip, an assumption rather than a verified historical execution cost.",
        f"- Shared segment-local warm-up: {WARMUP_BARS} observed candles after every gap greater than {GAP_TOLERANCE}. This exceeds finite RSI(14), volatility(60), and Donchian(20) windows and is five EMA 21 spans; it is a practical EMA convention, not exact elimination of recursive seed influence. The unused EMA 50 feature is not five-span-warmed by 105 bars.",
        "- Gaps strictly greater than 90 seconds reset every indicator; exactly 90 seconds remains in the same segment. This conservative rule also resets across weekends and other closures; no session calendar is applied.",
        "- All rules and parameters were fixed in advance of this comparison; no parameter search or training is performed. The script has no automatic strategy selector.",
        "- Period boundaries use the complete ordered candle timeline: 60% train, 20% validation, 20% test. Backtest candles are censored so each outcome closes strictly before its period end.",
        "- Accounting is disjoint: candidate events = pre-backtester warm-up exclusions + signals passed to backtester; passed signals = accepted trades + one backtester rejection reason. The report lists each stage explicitly.",
        "- Attribution is validated against the full ordered source dataset (row index, candle timestamp, strategy ID). Records lacking any required source field are counted as unavailable; they are not counted as verified.",
        "- The test period was included in the earlier EMA experiment report. Any test metrics below are previously accessed and descriptive only; they must not influence selection.",
        "- Prices are Dukascopy BID data. Spread, slippage, commissions, and fill quality are not historically modeled.",
        "",
        "## Dataset and period dates",
        "",
        f"- Candles: {len(market):,}; first UTC candle: {market['time'].iloc[0]}; last UTC candle: {market['time'].iloc[-1]}.",
        "",
        "| Period | Status | Start UTC (inclusive) | End UTC (exclusive) | Candles |",
        "|---|---|---|---|---:|",
    ]
    for period_name, period in periods.items():
        status = "previously accessed; descriptive only" if period_name == "test" else "selection analysis permitted"
        lines.append(
            f"| {period_name} | {status} | {period['start']} | {period['end']} | "
            f"{period['end_index'] - period['start_index']:,} |"
        )

    for period_name in ("train", "validation", "test"):
        status = " (PREVIOUSLY ACCESSED; DO NOT USE FOR SELECTION)" if period_name == "test" else ""
        lines += [
            "",
            f"## {period_name.title()} results{status}",
            "",
            "| Strategy | Candidate events | Pre-backtester warm-up exclusions (initial / post-gap) | Accepted trades | Backtester rejected (unavailable entry / unavailable exit / gap / overlap / other; total) | Win rate | Average net pips | Total net pips | Profit factor | Max drawdown (pips) | Accepted attribution verified / unavailable | Rejected attribution verified / unavailable |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for strategy_name in strategy_names:
            evaluation = evaluations[period_name][strategy_name]
            summary = evaluation["summary"]
            lines.append(
                f"| {strategy_name} | {evaluation['generated_signals_in_period']} | "
                f"{evaluation['rejected_warmup_initial']} / {evaluation['rejected_warmup_after_gap']} | "
                f"{summary['number_of_trades']} | "
                f"{_rejection_counts(evaluation)}; warm-up + backtester rejects = {evaluation['rejected_total']} | "
                f"{_metric(summary['win_rate'], percent=True)} | "
                f"{_metric(summary['average_net_pips'])} | "
                f"{_metric(summary['total_net_pips'])} | "
                f"{_metric(summary['profit_factor'])} | "
                f"{_metric(summary['max_drawdown_pips'])} | "
                f"{evaluation['accepted_attributed']} / {evaluation['accepted_attribution_unavailable']} | "
                f"{evaluation['rejected_attributed']} / {evaluation['rejected_attribution_unavailable']} |"
            )
        lines.append(
            "| No-trade benchmark | 0 | 0 / 0 | 0 | none | n/a | n/a | 0.0000 | n/a | 0.0000 | n/a | n/a |"
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "The tables are descriptive comparisons under the stated assumptions. They do not include a multiple-comparison-aware statistical test, confidence intervals, or evidence of a persistent edge. No statistical significance is claimed.",
        "",
        "The earlier EMA test result is already visible, so this report marks all test-period strategy metrics as previously accessed. Do not use these test metrics to select rules. A fresh, untouched period would be needed for a clean confirmatory evaluation after selection on train and validation.",
        "",
        "The exact CSV export invocation remains unavailable. BID-only candles and a fixed 1-pip deduction do not establish executable ask/bid fills, variable spread, commissions, or slippage. Validate these with timestamp-aligned bid/ask data and venue-specific execution records before interpreting the results as realistic trading outcomes.",
        "",
    ]
    return "\n".join(lines)


def _failure_category(error: Exception) -> str:
    if isinstance(error, sqlite3.Error):
        return "persistence_error"
    message = str(error).lower()
    if "at least" in message or "too small" in message or "no valid" in message:
        return "insufficient_data"
    if isinstance(error, ValueError):
        return "invalid_configuration"
    return "execution_error"


def run_comparison(
    report_path: Path | None = REPORT_PATH,
    *,
    retry_failed: bool = False,
    strategy_ids: Sequence[str] | None = None,
    store: ExperimentStore | None = None,
    context: Mapping[str, Any] | None = None,
    execution_owner: str | None = None,
    lease_seconds: float = 900,
    on_handle: Callable[[str, RunHandle], None] | None = None,
    write_report: bool = True,
    return_details: bool = False,
) -> Path | ComparisonRunResult:
    selected_ids = list(strategy_ids) if strategy_ids is not None else [
        spec.strategy_id for spec in list_strategies()
    ]
    selected_specs = validate_selected_strategy_ids(selected_ids)
    prepared = prepare_comparison_context() if context is None else dict(context)
    market = prepared["market"]
    periods = prepared["periods"]
    active_store = store or ExperimentStore()
    handles: dict[str, RunHandle] = {}
    names_by_id = {spec.strategy_id: spec.name for spec in selected_specs}
    names = [spec.name for spec in selected_specs]

    # Startup is request-by-request; no acquired handle is failed if a later
    # request needs explicit retry or otherwise fails during setup.
    for spec in selected_specs:
        handle = active_store.create_or_resume_run(
            **comparison_request_args(spec.strategy_id, prepared),
            retry_failed=retry_failed,
        )
        handles[spec.strategy_id] = handle
        if on_handle is not None:
            on_handle(spec.strategy_id, handle)

    claimed: list[str] = []
    try:
        evaluations: dict[str, dict[str, dict[str, Any]]] = {
            period: {} for period in ("train", "validation", "test")
        }
        active_names = [names_by_id[sid] for sid, handle in handles.items() if not handle.reused_completed]
        for strategy_id, handle in handles.items():
            if handle.reused_completed:
                for period in evaluations:
                    evaluations[period][names_by_id[strategy_id]] = active_store.evaluation_for_report(
                        handle.experiment_id, period,
                    )

        if active_names:
            stack = ExitStack()
            try:
                if execution_owner is not None:
                    # The lease is renewed through signal generation and persistence.
                    for handle in handles.values():
                        if not handle.reused_completed:
                            stack.enter_context(active_store.maintain_execution_claim(
                                handle.experiment_id, execution_owner, lease_seconds=lease_seconds,
                            ))
                            claimed.append(handle.experiment_id)
                try:
                    strategy_signals = generate_strategy_signals(market, strategy_names=active_names)
                except Exception as exc:
                    for strategy_id, handle in handles.items():
                        if not handle.reused_completed:
                            try:
                                active_store.record_failure(
                                    handle.experiment_id, _failure_category(exc), str(exc),
                                    claim_owner=execution_owner,
                                )
                            except (ValueError, ExecutionClaimLost):
                                pass
                    raise
                for strategy_id, handle in handles.items():
                    if handle.reused_completed:
                        continue
                    name = names_by_id[strategy_id]
                    try:
                        for period_name in ("train", "validation", "test"):
                            evaluations[period_name][name] = evaluate_period(
                                market, strategy_signals[name], periods[period_name],
                                execution_delay=EXECUTION_DELAY,
                                gap_tolerance=GAP_TOLERANCE,
                                transaction_cost_pips=TRANSACTION_COST_PIPS,
                                expected_signal_id_prefix=SIGNAL_ID_PREFIXES[name],
                            )
                        active_store.complete_run(
                            handle.experiment_id,
                            {period: evaluations[period][name] for period in ("train", "validation", "test")},
                            claim_owner=execution_owner,
                        )
                    except Exception as exc:
                        if handle.status == "running":
                            try:
                                active_store.record_failure(
                                    handle.experiment_id, _failure_category(exc), str(exc),
                                    claim_owner=execution_owner,
                                )
                            except (ValueError, ExecutionClaimLost):
                                pass
                        raise
            finally:
                stack.close()
    except Exception:
        for experiment_id in claimed:
            active_store.release_execution_claim(experiment_id, execution_owner or "")
        raise

    result = ComparisonRunResult(handles, periods, evaluations, market)
    if return_details or not write_report:
        return result
    if report_path is None:
        raise ValueError("report_path is required when write_report=True")
    report = render_report(periods, evaluations, market=market, strategy_names=names)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to: {report_path}")
    result.report_path = report_path
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Preserve a prior failed attempt and create a new experiment ID.",
    )
    args = parser.parse_args()
    run_comparison(args.report, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
