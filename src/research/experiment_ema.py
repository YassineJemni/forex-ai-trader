"""Controlled 9/21 EMA crossover baseline with chronological evaluation.

Run from the repository root with ``python -m src.research.experiment_ema``.
The strategy and 15-minute holding horizon are fixed. This script performs no
parameter search; the validation period is reported for protocol completeness.
Indicators reset after timestamp gaps greater than 90 seconds. The common
105-observed-candle warm-up is a practical five-span EMA convention and covers
the finite RSI/volatility windows, but cannot mathematically erase all recursive
EMA seed influence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from .backtester import run_backtest
from .data import DEFAULT_DATA_PATH, load_market_data, normalize_utc_timestamps
from .experiment_store import (
    ExperimentStore,
    PROJECT_ROOT as PROJECT_ROOT_FOR_STORE,
    code_provenance,
    dataset_metadata,
    utc_date_range,
)
from .features import (
    build_features,
    indicator_segment_ids,
    segment_bar_counts,
)
from .strategy_registry import get_strategy, validate_registry_configuration


FAST_EMA = 9
SLOW_EMA = 21
HOLDING_PERIOD = pd.Timedelta("15min")
BAR_DURATION = pd.Timedelta("1min")
TIMESTAMP_CONVENTION = "bar_open"
WARMUP_BARS = 5 * SLOW_EMA
# For the current three strategies, 105 bars exceeds the finite RSI(14),
# volatility(60), and Donchian(20) windows and is five slow-EMA spans. The
# generic ema_50 feature is not used by these rules and is not five-span-warmed.
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PIP_SIZE = 0.0001
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "experiment_ema.md"


def validate_signal_attribution(
    records: pd.DataFrame,
    source_market: pd.DataFrame,
    *,
    expected_signal_id_prefix: str | None = None,
) -> tuple[int, int]:
    """Validate source row, source candle time, and ID against the full source data.

    Returns (verified, unavailable). Incomplete attribution is counted as
    unavailable; inconsistent attribution raises ValueError. ``source_market``
    must be the same ordered dataset used to create source-row indices.
    """
    required = {"signal_id", "source_row_index", "source_candle_timestamp"}
    if not required.issubset(records.columns):
        return 0, len(records)
    def utc_series(values: pd.Series) -> pd.Series:
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            return values.dt.tz_convert("UTC")
        return normalize_utc_timestamps(values)

    source_times = utc_series(source_market["time"])
    normalized_record_times = utc_series(records["source_candle_timestamp"])
    verified = unavailable = 0
    for signal_id, source_index, normalized in zip(
        records["signal_id"], records["source_row_index"], normalized_record_times
    ):
        if (pd.isna(signal_id) or pd.isna(source_index)
                or pd.isna(normalized)):
            unavailable += 1
            continue
        if (isinstance(source_index, (bool, np.bool_))
                or not isinstance(source_index, (int, np.integer))):
            raise ValueError(f"invalid source_row_index in attribution: {source_index!r}")
        index = int(source_index)
        if index < 0 or index >= len(source_market):
            raise ValueError(f"source_row_index outside source dataset: {index}")
        if normalized != source_times.iloc[index]:
            raise ValueError(
                f"source row/timestamp mismatch at row {index}: {normalized!r}"
            )
        expected_id = f"{expected_signal_id_prefix or str(signal_id).split(':', 1)[0]}:{normalized.isoformat()}"
        if str(signal_id) != expected_id:
            raise ValueError(
                f"signal_id does not match source strategy/timestamp: {signal_id!r}"
            )
        verified += 1
    return verified, unavailable


def generate_ema_crossover_signals(
    market_data: pd.DataFrame,
    *,
    bar_duration: str | pd.Timedelta = BAR_DURATION,
) -> pd.DataFrame:
    """Generate close-time signals using current and prior candles only."""
    features = build_features(market_data)
    segments = indicator_segment_ids(features["time"])
    bars_in_segment = segment_bar_counts(segments)
    fast = features["ema_9"]
    slow = features["ema_21"]

    spread = fast - slow
    previous = spread.shift(1)
    same_segment = segments.eq(segments.shift(1))
    long_cross = same_segment & (previous <= 0) & (spread > 0)
    short_cross = same_segment & (previous >= 0) & (spread < 0)
    direction = np.select([long_cross, short_cross], [1, -1], default=0)
    selected = np.flatnonzero(direction != 0)
    duration = pd.Timedelta(bar_duration)
    if duration <= pd.Timedelta(0):
        raise ValueError("bar_duration must be positive")

    source_times = features["time"].iloc[selected].reset_index(drop=True)
    return pd.DataFrame({
        "signal_id": [f"ema_9_21:{timestamp.isoformat()}" for timestamp in source_times],
        "decision_timestamp": source_times + duration,
        "direction": np.where(direction[selected] > 0, "long", "short"),
        "source_row_index": selected,
        "source_index": selected,  # Backward-compatible alias.
        "source_candle_timestamp": source_times,
        "indicator_segment_id": segments.iloc[selected].to_numpy(),
        "segment_bar_count": bars_in_segment.iloc[selected].to_numpy(),
        "indicator_warmup_complete": (
            bars_in_segment.iloc[selected].to_numpy() >= WARMUP_BARS
        ),
    })


def timeline_periods(
    market: pd.DataFrame,
    *,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    bar_duration: str | pd.Timedelta = BAR_DURATION,
) -> dict[str, dict[str, Any]]:
    """Partition the complete candle timeline into chronological intervals."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("period fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be less than one")
    if len(market) < 3:
        raise ValueError("at least three candles are required to define periods")
    times = pd.DatetimeIndex(market["time"])
    if times.tz is None:
        raise ValueError("market timestamps must be timezone-aware")
    train_end = int(len(times) * train_fraction)
    validation_end = int(len(times) * (train_fraction + validation_fraction))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(times):
        raise ValueError("timeline is too small for nonempty train/validation/test periods")
    end_of_data = times[-1] + pd.Timedelta(bar_duration) + pd.Timedelta(1, "ns")
    boundaries = [times[0], times[train_end], times[validation_end], end_of_data]
    names = ["train", "validation", "test"]
    return {
        name: {"start": boundaries[i], "end": boundaries[i + 1],
               "start_index": [0, train_end, validation_end][i],
               "end_index": [train_end, validation_end, len(times)][i]}
        for i, name in enumerate(names)
    }


def split_metadata(periods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Serializable exact split definition used in each experiment request."""
    return {
        "policy": "ordered_complete_candle_timeline",
        "fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION,
        },
        "periods": {
            name: {
                "start_utc": pd.Timestamp(value["start"]).isoformat(),
                "end_utc": pd.Timestamp(value["end"]).isoformat(),
                "start_index": int(value["start_index"]),
                "end_index": int(value["end_index"]),
            }
            for name, value in periods.items()
        },
    }


def validate_ema_registry_configuration() -> None:
    validate_registry_configuration({
        "ema_crossover": {
            "fast_ema": FAST_EMA,
            "slow_ema": SLOW_EMA,
            "holding_minutes": int(HOLDING_PERIOD.total_seconds() / 60),
            "warmup_bars": WARMUP_BARS,
        }
    })


def evaluate_period(
    market: pd.DataFrame,
    signals: pd.DataFrame,
    period: dict[str, Any],
    *,
    execution_delay: str | pd.Timedelta,
    gap_tolerance: str | pd.Timedelta,
    transaction_cost_pips: float,
    expected_signal_id_prefix: str | None = None,
) -> dict[str, Any]:
    """Backtest signals in one timeline period, censoring at its end boundary.

    Restrict the backtest candles to those closing strictly before the period
    end. The shared backtester then rejects unavailable exits at that cutoff;
    no trade outcome can use a candle from the next period.
    """
    start, end = period["start"], period["end"]
    period_signals = signals.loc[
        (signals["decision_timestamp"] >= start)
        & (signals["decision_timestamp"] < end)
    ].copy()
    candidate_attribution_verified, candidate_attribution_unavailable = (
        validate_signal_attribution(
            period_signals, market,
            expected_signal_id_prefix=expected_signal_id_prefix,
        )
    )
    # Candidate events before the segment-local warm-up remain attributable,
    # but are not sent to the backtester.
    if "indicator_warmup_complete" in period_signals:
        warmup_complete = period_signals["indicator_warmup_complete"].astype(bool)
    else:
        source_index_column = (
            "source_row_index" if "source_row_index" in period_signals else "source_index"
        )
        warmup_complete = period_signals[source_index_column] >= WARMUP_BARS
    rejected_warmup = period_signals.loc[~warmup_complete].copy()
    if "indicator_segment_id" in rejected_warmup:
        after_gap = rejected_warmup["indicator_segment_id"] > 0
    else:
        after_gap = pd.Series(False, index=rejected_warmup.index)
    rejected_warmup["rejection_reason"] = np.where(
        after_gap, "indicator_warmup_after_gap", "indicator_warmup_initial"
    )
    period_signals = period_signals.loc[warmup_complete].copy()
    # Require the final bar close used by the backtester to be strictly before
    # the split boundary. The last test period ends one nanosecond past data end.
    available_market = market.loc[
        market["time"] + BAR_DURATION < end
    ].copy()
    signal_columns = [
        column for column in (
            "decision_timestamp", "direction", "signal_id",
            "source_row_index", "source_candle_timestamp",
        ) if column in period_signals
    ]
    result = run_backtest(
        period_signals[signal_columns],
        available_market,
        holding_period=HOLDING_PERIOD,
        timestamp_convention=TIMESTAMP_CONVENTION,
        execution_delay=execution_delay,
        allow_overlapping=False,
        transaction_cost_pips=transaction_cost_pips,
        pip_size=PIP_SIZE,
        bar_duration=BAR_DURATION,
        gap_tolerance=gap_tolerance,
    )
    accepted_attributed, accepted_attribution_unavailable = validate_signal_attribution(
        result["trades"], market,
        expected_signal_id_prefix=expected_signal_id_prefix,
    )
    rejected_records = pd.concat(
        [rejected_warmup, result["rejected_signals"]],
        ignore_index=True,
        sort=False,
    )
    rejected_attributed, rejected_attribution_unavailable = validate_signal_attribution(
        rejected_records, market,
        expected_signal_id_prefix=expected_signal_id_prefix,
    )
    rejected_total = len(rejected_warmup) + sum(result["rejected"].values())
    if len(period_signals) + len(rejected_warmup) != int(
        ((signals["decision_timestamp"] >= start)
         & (signals["decision_timestamp"] < end)).sum()
    ):
        raise AssertionError("candidate/warm-up accounting does not reconcile")
    if len(result["trades"]) + sum(result["rejected"].values()) != len(period_signals):
        raise AssertionError("backtester accepted/rejected accounting does not reconcile")
    return {
        **result,
        "generated_signals_in_period": int(
            ((signals["decision_timestamp"] >= start)
             & (signals["decision_timestamp"] < end)).sum()
        ),
        "signals_after_warmup": len(period_signals),
        "rejected_warmup": len(rejected_warmup),
        "rejected_warmup_initial": int((~after_gap).sum()),
        "rejected_warmup_after_gap": int(after_gap.sum()),
        "rejected_signals": rejected_records,
        "rejected_at_period_end": result["rejected"]["unavailable_exit"],
        "rejected_total": rejected_total,
        "accepted_attributed": accepted_attributed,
        "rejected_attributed": rejected_attributed,
        "candidate_attribution_verified": candidate_attribution_verified,
        "candidate_attribution_unavailable": candidate_attribution_unavailable,
        "accepted_attribution_unavailable": accepted_attribution_unavailable,
        "rejected_attribution_unavailable": rejected_attribution_unavailable,
    }


def _format_metric(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)) and np.isinf(value):
        return "∞"
    if percent:
        return f"{float(value) * 100:.2f}%"
    return f"{float(value):.4f}"


def render_report(
    periods: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    *,
    execution_delay: str,
    gap_tolerance: str,
    transaction_cost_pips: float,
    market_rows: int,
    first_time: pd.Timestamp,
    last_time: pd.Timestamp,
) -> str:
    lines = [
        "# Controlled EMA Crossover Baseline",
        "",
        "Results below were produced with segment-local indicator resets at gaps greater than 90 seconds.",
        "",
        "This is a fixed research experiment, not a profitability claim.",
        "",
        "## Configuration and assumptions",
        "",
        f"- Strategy: EMA {FAST_EMA}/{SLOW_EMA} crossover; long on at/below-to-above, short on at/above-to-below.",
        f"- Holding horizon: {HOLDING_PERIOD}; bar duration: {BAR_DURATION}.",
        f"- Timestamp convention: `{TIMESTAMP_CONVENTION}`; signal decision at signal candle close.",
        f"- Execution delay: `{execution_delay}`; entry at first eligible candle open strictly after decision plus delay.",
        f"- Gap tolerance: `{gap_tolerance}`; overlap: disabled.",
        f"- Fixed transaction cost: {transaction_cost_pips:.4f} pips per round trip. This is an assumption, not a verified historical spread or execution cost.",
        f"- Indicator warm-up: {WARMUP_BARS} observed candles within each segment. It exceeds the finite RSI(14), volatility(60), and Donchian(20) windows and is five spans of the current EMA 21; for recursive EMA it is a practical convention, not exact elimination of seed influence. The unused EMA 50 feature is not five-span-warmed by 105 bars.",
        "- Gaps greater than 90 seconds reset indicator state; exactly 90 seconds remains in the same segment. This conservative rule applies to weekends and other closures too; no session calendar is used.",
        "- No parameters were selected or tuned. Train, validation, and test use the same fixed strategy.",
        "- Test-period output is included only as a previously accessed descriptive evaluation; it is not a selection criterion.",
        "- Candle times are UTC and use the evidence-supported Dukascopy M1 `bar_open` convention. Exact CSV export invocation is unavailable.",
        "- Data contains Dukascopy BID prices; realistic spread, slippage, commissions, and fill quality are not modeled.",
        "- Each period's backtest data is censored so exit candles close strictly before that period's end. Signals whose holding outcome is incomplete at the cutoff are rejected by the backtester.",
        "",
        "## Dataset and periods",
        "",
        f"- Candles: {market_rows:,}; first UTC candle: {first_time}; last UTC candle: {last_time}.",
        "- Period boundaries are computed from the complete ordered candle timeline (60% / 20% / 20%), not from signal counts.",
        "",
        "| Period | Status | Start UTC (inclusive) | End UTC (exclusive) | Timeline candles | Candidate events | Passed warm-up | Initial warm-up excluded | Post-gap warm-up excluded | Backtester rejected | Total rejected |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, period in periods.items():
        evaluation = evaluations[name]
        status = "previously accessed; descriptive only" if name == "test" else "development period"
        lines.append(
            f"| {name} | {status} | {period['start']} | {period['end']} | "
            f"{period['end_index'] - period['start_index']:,} | "
            f"{evaluation['generated_signals_in_period']:,} | {evaluation['signals_after_warmup']:,} | "
            f"{evaluation['rejected_warmup_initial']:,} | {evaluation['rejected_warmup_after_gap']:,} | "
            f"{sum(evaluation['rejected'].values()):,} | {evaluation['rejected_total']:,} |"
        )
    lines += [
        "",
        "Warm-up columns count candidate events excluded before the backtester, not bars removed. Zero initial-segment warm-up rejections means no candidate event was rejected in that initial warm-up; it does not mean the first 105 bars were already warmed. Backtester gap rejections cover gaps from decision through exit.",
        "",
        "## Backtester rejection breakdown",
        "",
        "Reason columns count signals passed to the backtester. Total rejected also includes the initial and post-gap warm-up exclusions shown above.",
        "",
        "| Period | Invalid decision | Invalid direction | Unavailable entry | Invalid entry price | Unavailable exit | Invalid exit price | Gap exceeded | Overlap skipped | Total rejected | Verified / unavailable attribution (rejected) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    reason_columns = [
        "invalid_decision_timestamp", "invalid_direction", "unavailable_entry",
        "invalid_entry_price", "unavailable_exit", "invalid_exit_price",
        "gap_exceeds_tolerance", "overlap_skipped",
    ]
    for name, evaluation in evaluations.items():
        counts = evaluation["rejected"]
        lines.append(
            f"| {name} | " + " | ".join(str(counts[key]) for key in reason_columns)
            + f" | {evaluation['rejected_total']} | {evaluation['rejected_attributed']} / {evaluation['rejected_attribution_unavailable']} |"
        )
    lines += [
        "",
        "## Results",
        "",
        "| Period | Baseline | Trades | Win rate | Total net pips | Average net pips | Profit factor | Max drawdown (pips) | Verified / unavailable attribution (accepted) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, evaluation in evaluations.items():
        summary = evaluation["summary"]
        display_name = (
            "test (previously accessed; descriptive only)" if name == "test" else name
        )
        lines.append(
            f"| {display_name} | EMA 9/21 | {summary['number_of_trades']} | "
            f"{_format_metric(summary['win_rate'], percent=True)} | "
            f"{_format_metric(summary['total_net_pips'])} | "
            f"{_format_metric(summary['average_net_pips'])} | "
            f"{_format_metric(summary['profit_factor'])} | "
            f"{_format_metric(summary['max_drawdown_pips'])} | {evaluation['accepted_attributed']} / {evaluation['accepted_attribution_unavailable']} |"
        )
        lines.append(
            f"| {display_name} | No-trade | 0 | n/a | 0.0000 | n/a | n/a | 0.0000 | n/a |"
        )
    lines += [
        "",
        "No-trade metrics that require trades are not applicable. Break-even trades count in the backtest win-rate denominator; profit factor is infinite with gains and no losses, and undefined with no gains or losses.",
        "Attribution verified means source row, UTC candle timestamp, and strategy-prefixed signal ID matched the full ordered source dataset. Records without all required source fields are counted unavailable, not verified.",
        "",
        "This report describes the output of one fixed historical experiment. It does not establish future performance or profitability.",
        "",
    ]
    return "\n".join(lines)


def run_experiment(
    *,
    execution_delay: str = "0s",
    gap_tolerance: str = "90s",
    transaction_cost_pips: float = 1.0,
    report_path: Path = REPORT_PATH,
    retry_failed: bool = False,
) -> Path:
    validate_ema_registry_configuration()
    market = load_market_data()
    periods = timeline_periods(market)
    split = split_metadata(periods)
    fingerprint, identity = dataset_metadata(DEFAULT_DATA_PATH, market)
    date_start, date_end = utc_date_range(market)
    spec = get_strategy("ema_crossover")
    store = ExperimentStore()
    handle = store.create_or_resume_run(
        strategy=spec,
        parameters=spec.parameters,
        dataset_fingerprint=fingerprint,
        dataset_identity=identity,
        date_start_utc=date_start,
        date_end_utc=date_end,
        execution_assumptions={
            "experiment_kind": "ema_baseline",
            "timestamp_convention": TIMESTAMP_CONVENTION,
            "bar_duration": str(BAR_DURATION),
            "holding_period": str(HOLDING_PERIOD),
            "execution_delay": str(execution_delay),
            "allow_overlapping": False,
            "transaction_cost_pips_round_trip": transaction_cost_pips,
            "pip_size": PIP_SIZE,
        },
        gap_policy={"tolerance": str(gap_tolerance),
                    "indicator_reset_above": "90s", "exact_boundary_resets": False},
        software=code_provenance(PROJECT_ROOT_FOR_STORE),
        data_quality=dict(market.attrs.get("load_report", {})),
        split_metadata=split,
        retry_failed=retry_failed,
    )
    if handle.reused_completed:
        stored_run = store.get_run(handle.experiment_id)
        periods = {
            name: {
                "start": pd.Timestamp(values["start_utc"]),
                "end": pd.Timestamp(values["end_utc"]),
                "start_index": values["start_index"],
                "end_index": values["end_index"],
            }
            for name in ("train", "validation", "test")
            for values in (stored_run["split_metadata"]["periods"][name],)
        }
        evaluations = {
            name: store.evaluation_for_report(handle.experiment_id, name)
            for name in ("train", "validation", "test")
        }
    else:
        try:
            signals = generate_ema_crossover_signals(market)
            evaluations = {
                name: evaluate_period(
                    market,
                    signals,
                    period,
                    execution_delay=execution_delay,
                    gap_tolerance=gap_tolerance,
                    transaction_cost_pips=transaction_cost_pips,
                    expected_signal_id_prefix="ema_9_21",
                )
                for name, period in periods.items()
            }
            store.complete_run(handle.experiment_id, evaluations)
        except Exception as exc:
            if handle.status == "running":
                message = str(exc).lower()
                category = (
                    "persistence_error" if isinstance(exc, sqlite3.Error)
                    else
                    "insufficient_data" if "at least" in message or "too small" in message
                    else "invalid_configuration" if isinstance(exc, ValueError)
                    else "execution_error"
                )
                try:
                    store.record_failure(handle.experiment_id, category, str(exc))
                except ValueError:
                    pass
            raise
    report = render_report(
        periods,
        evaluations,
        execution_delay=execution_delay,
        gap_tolerance=gap_tolerance,
        transaction_cost_pips=transaction_cost_pips,
        market_rows=len(market),
        first_time=market["time"].iloc[0],
        last_time=market["time"].iloc[-1],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-delay", default="0s")
    parser.add_argument("--gap-tolerance", default="90s")
    parser.add_argument(
        "--transaction-cost-pips", type=float, default=1.0,
        help="Assumed fixed round-trip cost in pips; not a verified historical cost.",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Preserve a prior failed attempt and create a new experiment ID.",
    )
    args = parser.parse_args()
    run_experiment(
        execution_delay=args.execution_delay,
        gap_tolerance=args.gap_tolerance,
        transaction_cost_pips=args.transaction_cost_pips,
        report_path=args.report,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
