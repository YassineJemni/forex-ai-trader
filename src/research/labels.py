"""Causal forward labels and purged chronological splits.

Rules and assumptions
---------------------
* Features are taken from the signal candle's row and use only its OHLC values
  and prior rows. The decision is made at that candle's close.
* ``timestamp_convention`` is required: ``bar_open`` means ``time`` marks the
  candle start; ``bar_close`` means it marks the candle end. For this dataset,
  Dukascopy M1 candles use start timestamps, and the CSV retains the parquet's
  UTC clock values after its timezone marker was removed.
* Search candle open times for the first open strictly later than the decision
  timestamp plus ``execution_delay``. Thus an open sharing the decision time
  cannot be used; zero delay still requires a strictly later timestamp. The
  requested horizon starts at this actual entry timestamp.
* Exit at the close of the first candle ending at or after entry plus horizon.
  With minute candles, the realized interval may be up to one bar longer.
* Reject samples with incomplete features, unavailable entry/exit, a gap above
  ``gap_tolerance`` between any candles from decision through exit, or an exact
  flat outcome. Target is 1 for exit close > entry open, else 0.

The loader localizes the source CSV's naive clock values as UTC. Timezone-aware
inputs are converted to UTC; mixed naive/aware inputs are rejected. The exact
CSV export invocation is unavailable, so the convention is based on the local
Dukascopy candle documentation and exact CSV/parquet content match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    load_market_data,
    normalize_utc_timestamps,
    validate_m1_cadence,
    validate_ohlc_prices,
)
from .execution_models import MarketDataMetadata, prepare_bid_ask_inputs
from .experiment_store import code_provenance, fingerprint_market_frame
from .features import build_features


SAMPLE_COLUMNS = [
    "decision_timestamp", "entry_timestamp", "exit_timestamp",
    "entry_price", "exit_price", "target", "future_outcome_end_time",
]

MID_RETURN_HORIZONS = ("1min", "5min", "15min", "60min")
MID_RETURN_FEATURE_COLUMNS = (
    "return_1", "return_5", "return_15", "return_60",
    "ema_gap", "ema_long_gap", "rsi_14",
    "volatility_15", "volatility_60",
    "candle_range", "candle_body", "time_sin", "time_cos",
    "spread_mid_ratio", "spread_mid_ratio_mean_15",
    "spread_mid_ratio_std_15",
)
MID_RETURN_TRAINING_ROWS = 404_889
MID_RETURN_BOUNDARY_UTC = "2026-02-03T18:36:00Z"


@dataclass(frozen=True)
class TrainingBoundarySpec:
    """Verified position of a training boundary on the original candle index."""

    training_rows: int
    boundary_timestamp_utc: str

    def __post_init__(self) -> None:
        if (isinstance(self.training_rows, bool) or not isinstance(self.training_rows, int)
                or self.training_rows < 1):
            raise ValueError("training boundary row counts are invalid")
        boundary = pd.Timestamp(self.boundary_timestamp_utc)
        if boundary.tzinfo is None or boundary.tz_convert("UTC") != boundary:
            raise ValueError("training boundary timestamp must be UTC-aware")
        if boundary.isoformat().replace("+00:00", "Z") != self.boundary_timestamp_utc:
            raise ValueError("training boundary timestamp must use canonical UTC ISO format")


FROZEN_TRAINING_BOUNDARY = TrainingBoundarySpec(
    training_rows=MID_RETURN_TRAINING_ROWS,
    boundary_timestamp_utc=MID_RETURN_BOUNDARY_UTC,
)


def _verified_training_timeline(
    timestamps: pd.Series,
    spec: TrainingBoundarySpec,
) -> pd.Timestamp:
    times = normalize_utc_timestamps(timestamps)
    ordered = times.sort_values(kind="stable").reset_index(drop=True)
    if len(ordered) <= spec.training_rows:
        raise ValueError(
            "cannot verify frozen training boundary: original timeline ends "
            f"before index {spec.training_rows}"
        )
    training_timeline = ordered.iloc[:spec.training_rows + 1]
    if training_timeline.isna().any():
        raise ValueError(
            "cannot verify frozen training boundary: invalid timestamp before boundary"
        )
    if training_timeline.duplicated().any():
        raise ValueError(
            "cannot verify frozen training boundary: duplicate timestamp before boundary"
        )
    try:
        validate_m1_cadence(training_timeline)
    except ValueError as exc:
        raise ValueError(f"cannot verify frozen training boundary: {exc}") from exc
    boundary = pd.Timestamp(spec.boundary_timestamp_utc)
    observed = ordered.iloc[spec.training_rows]
    if observed != boundary:
        raise ValueError(
            "cannot verify frozen training boundary: timestamp at original "
            f"index {spec.training_rows} is {observed.isoformat()}, "
            f"expected {boundary.isoformat()}"
        )
    return boundary


def build_mid_return_samples(
    bid_data: pd.DataFrame,
    ask_data: pd.DataFrame,
    *,
    bid_metadata: MarketDataMetadata,
    ask_metadata: MarketDataMetadata,
    boundary_spec: TrainingBoundarySpec = FROZEN_TRAINING_BOUNDARY,
    horizons: tuple[str, ...] = MID_RETURN_HORIZONS,
    timestamp_convention: str = "bar_open",
    bar_duration: str | pd.Timedelta = "1min",
    gap_tolerance: str | pd.Timedelta = "90s",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build timestamp-exact continuous BID/ASK midpoint log-return labels.

    The original BID timeline must match ``boundary_spec`` exactly. Prices are
    then restricted to timestamps strictly before the verified boundary before
    validation, features, or labels are constructed. Entry is the exact next
    M1 open after the close-time decision. The intended exit is the exact
    candle close at ``entry + horizon``; missing timestamps are rejected.
    """
    if timestamp_convention != "bar_open":
        raise ValueError("continuous mid-return labels currently require bar_open timestamps")
    duration = pd.Timedelta(bar_duration)
    tolerance = pd.Timedelta(gap_tolerance)
    if duration != pd.Timedelta("1min"):
        raise ValueError("continuous mid-return labels require one-minute bars")
    if tolerance < duration:
        raise ValueError("gap_tolerance must be at least one bar duration")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be a nonempty sequence without duplicates")
    horizon_deltas = {value: pd.Timedelta(value) for value in horizons}
    if any(value <= pd.Timedelta(0) for value in horizon_deltas.values()):
        raise ValueError("horizons must be positive durations")

    if not isinstance(bid_data, pd.DataFrame) or not isinstance(ask_data, pd.DataFrame):
        raise TypeError("BID and ASK inputs must be pandas DataFrames")
    if "time" not in bid_data.columns or "time" not in ask_data.columns:
        raise ValueError("BID and ASK inputs must contain time columns")

    boundary = _verified_training_timeline(
        bid_data["time"], boundary_spec,
    )
    bid_times = normalize_utc_timestamps(bid_data["time"])
    ask_times = normalize_utc_timestamps(ask_data["time"])
    if ask_times.isna().any() or ask_times.duplicated().any():
        raise ValueError("cannot verify BID/ASK training alignment: invalid or duplicate ASK timestamp")
    # Only timestamps are used from the post-boundary portion; no post-boundary
    # price or feature values enter validation, feature construction, or labels.
    bid = bid_data.loc[bid_times < boundary].copy()
    bid["time"] = bid_times.loc[bid_times < boundary]
    bid = bid.sort_values("time", kind="stable").reset_index(drop=True)
    ask_mask = ask_times < boundary
    ask = ask_data.loc[ask_mask].copy()
    ask["time"] = ask_times.loc[ask_mask]
    ask = ask.sort_values("time", kind="stable").reset_index(drop=True)
    if len(bid) != boundary_spec.training_rows:
        raise ValueError(
            "cannot freeze training data: BID rows strictly before boundary "
            f"are {len(bid)}, expected {boundary_spec.training_rows}"
        )

    bid, ask, alignment = prepare_bid_ask_inputs(
        bid, ask,
        bid_metadata=bid_metadata,
        ask_metadata=ask_metadata,
        timestamp_convention=timestamp_convention,
        bar_duration=duration,
    )
    # The timestamp merge is exact and one-to-one. Missing ASK observations
    # remain missing and are reported rather than filled.
    paired = bid.merge(
        ask[["time", "open", "close"]].rename(columns={
            "open": "ask_open", "close": "ask_close",
        }),
        on="time", how="left", validate="one_to_one",
    )
    if ((paired["ask_open"].notna() & (paired["ask_open"] < paired["open"]))
            | (paired["ask_close"].notna() & (paired["ask_close"] < paired["close"]))).any():
        raise ValueError("training BID/ASK data contains crossed open or close quotes")

    featured = build_features(bid, gap_tolerance=tolerance)
    mid_close = (paired["close"] + paired["ask_close"]) / 2.0
    spread_ratio = (paired["ask_close"] - paired["close"]) / mid_close
    segments = (featured["time"].diff() > tolerance).cumsum()
    featured["spread_mid_ratio"] = spread_ratio
    featured["spread_mid_ratio_mean_15"] = spread_ratio.groupby(
        segments, sort=False,
    ).transform(lambda values: values.rolling(15, min_periods=15).mean())
    featured["spread_mid_ratio_std_15"] = spread_ratio.groupby(
        segments, sort=False,
    ).transform(lambda values: values.rolling(15, min_periods=15).std())

    bid_times_index = pd.DatetimeIndex(bid["time"])
    ask_times_index = pd.DatetimeIndex(ask["time"])
    bid_position = {stamp: index for index, stamp in enumerate(bid_times_index)}
    ask_position = {stamp: index for index, stamp in enumerate(ask_times_index)}
    bid_open = bid["open"].to_numpy(dtype=float)
    bid_close = bid["close"].to_numpy(dtype=float)
    ask_open = ask["open"].to_numpy(dtype=float)
    ask_close = ask["close"].to_numpy(dtype=float)

    feature_columns = list(MID_RETURN_FEATURE_COLUMNS)
    records: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for source_index, source_time in enumerate(bid_times_index):
        decision = source_time + duration
        if decision >= boundary:
            continue
        expected_entry = decision + duration
        for horizon, horizon_delta in horizon_deltas.items():
            expected_exit_close = expected_entry + horizon_delta
            expected_exit_open = expected_exit_close - duration
            reason: str | None = None
            entry_i = bid_position.get(expected_entry)
            exit_i = bid_position.get(expected_exit_open)

            if expected_entry >= boundary:
                reason = "entry_at_or_after_training_boundary"
            elif entry_i is None:
                reason = "entry_timestamp_missing"
            elif expected_entry not in ask_position:
                reason = "unmatched_entry_quote"
            elif expected_exit_open >= boundary or expected_exit_close >= boundary:
                reason = "exit_at_or_after_training_boundary"
            elif exit_i is None:
                reason = "exact_exit_timestamp_missing"
            elif expected_exit_open not in ask_position:
                reason = "unmatched_exact_exit_quote"
            else:
                outcome_times = pd.Series(
                    bid_times_index[source_index:exit_i + 1],
                )
                if outcome_times.diff().gt(tolerance).any():
                    reason = "gap_exceeds_tolerance"
                elif ask_open[ask_position[expected_entry]] < bid_open[entry_i]:
                    reason = "crossed_entry_quote"
                elif ask_close[ask_position[expected_exit_open]] < bid_close[exit_i]:
                    reason = "crossed_exit_quote"

            entry_mid: float | None = None
            exit_mid: float | None = None
            log_return: float | None = None
            if reason is None:
                entry_mid = (
                    bid_open[entry_i] + ask_open[ask_position[expected_entry]]
                ) / 2.0
                exit_mid = (
                    bid_close[exit_i] + ask_close[ask_position[expected_exit_open]]
                ) / 2.0
                if entry_mid <= 0 or exit_mid <= 0:
                    reason = "invalid_mid_price"
                else:
                    log_return = float(np.log(exit_mid / entry_mid))

            record: dict[str, Any] = {
                "decision_timestamp": decision,
                "entry_timestamp": expected_entry,
                "exit_timestamp": expected_exit_close,
                "horizon": horizon,
                "entry_mid": entry_mid,
                "exit_mid": exit_mid,
                "forward_log_return": log_return,
                "accepted": reason is None,
                "rejection_reason": reason,
            }
            for name in feature_columns:
                record[name] = featured.iloc[source_index][name]
            records.append(record)
            if reason is not None:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    output_columns = [
        "decision_timestamp", "entry_timestamp", "exit_timestamp", "horizon",
        "entry_mid", "exit_mid", "forward_log_return", "accepted",
        "rejection_reason", *feature_columns,
    ]
    samples = pd.DataFrame.from_records(records, columns=output_columns)
    training_fingerprints = []
    for frame, side in ((bid, "BID"), (ask, "ASK")):
        fingerprint, _ = fingerprint_market_frame(
            frame,
            quality_policy={
                "policy_id": "mission4_training_frame_v1",
                "market_side": side,
                "training_boundary_utc": boundary.isoformat(),
            },
        )
        training_fingerprints.append(fingerprint)
    training_fingerprint = hashlib.sha256(
        "".join(training_fingerprints).encode("ascii")
    )
    audit = {
        "boundary": {
            **asdict(boundary_spec),
            "verified_index": boundary_spec.training_rows,
            "verified_timestamp_utc": boundary.isoformat(),
            "post_boundary_prices_used": False,
        },
        "training_bid_rows": int(len(bid)),
        "training_ask_rows": int(len(ask)),
        "bid_ask_alignment": alignment,
        "accepted": int(samples["accepted"].sum()) if not samples.empty else 0,
        "rejected": int((~samples["accepted"]).sum()) if not samples.empty else 0,
        "rejected_by_reason": rejection_counts,
        "feature_columns": feature_columns,
        "horizons": list(horizons),
        "training_data_fingerprint_sha256": training_fingerprint.hexdigest(),
        "software_provenance": code_provenance(),
        "label_policy": "next_exact_M1_open; exact_entry_plus_elapsed_horizon_close; midpoint_log_return",
    }
    return samples, audit


def build_labeled_samples(
    data: pd.DataFrame | None = None,
    *,
    horizon: str | pd.Timedelta = "15min",
    timestamp_convention: str,
    execution_delay: str | pd.Timedelta = "0s",
    bar_duration: str | pd.Timedelta = "1min",
    gap_tolerance: str | pd.Timedelta = "90s",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return causal features with a binary first-eligible-open target.

    Target is 1 if exit close > entry open and 0 if lower. Exact flats are
    rejected. Incomplete features, unavailable event bars, and any timestamp
    gap above ``gap_tolerance`` from decision through exit are also rejected.
    ``execution_delay`` is a minimum wait after the close decision; entry is
    the first candle open strictly later than decision time plus that delay.
    The result includes the event columns in :data:`SAMPLE_COLUMNS`.
    """
    if timestamp_convention not in {"bar_open", "bar_close"}:
        raise ValueError("timestamp_convention must be 'bar_open' or 'bar_close'")
    horizon_delta = pd.Timedelta(horizon)
    delay_delta = pd.Timedelta(execution_delay)
    bar_delta = pd.Timedelta(bar_duration)
    tolerance = pd.Timedelta(gap_tolerance)
    if horizon_delta <= pd.Timedelta(0) or bar_delta <= pd.Timedelta(0):
        raise ValueError("horizon and bar_duration must be positive")
    if tolerance < pd.Timedelta(0):
        raise ValueError("gap_tolerance cannot be negative")
    if delay_delta < pd.Timedelta(0):
        raise ValueError("execution_delay cannot be negative")

    market = load_market_data() if data is None else data.copy()
    source_load_report = dict(market.attrs.get("load_report", {}))
    input_rows = len(market)
    if not {"time", "open", "high", "low", "close"}.issubset(market.columns):
        raise ValueError("data must contain time and OHLC columns")
    market["time"] = normalize_utc_timestamps(market["time"])
    invalid_timestamp = market["time"].isna()
    dropped_invalid_timestamps = int(invalid_timestamp.sum())
    market = market.loc[~invalid_timestamp].copy()
    market = validate_ohlc_prices(market)
    duplicate_timestamps = int(market["time"].duplicated(keep="last").sum())
    market = (market.sort_values("time")
              .drop_duplicates("time", keep="last")
              .reset_index(drop=True))
    if market.empty:
        raise ValueError("No valid candles available")

    featured = build_features(market)
    raw_columns = {"time", "open", "high", "low", "close", "volume"}
    feature_columns = [c for c in featured.columns if c not in raw_columns]
    feature_ok = featured[feature_columns].notna().all(axis=1).to_numpy()

    times = pd.DatetimeIndex(market["time"]).to_numpy(dtype="datetime64[ns]")
    n = len(times)
    indices = np.arange(n)
    decision_times = times.copy()
    if timestamp_convention == "bar_open":
        decision_times += bar_delta.to_timedelta64()
        candle_open_times = times
        candle_end_times = times + bar_delta.to_timedelta64()
    else:
        candle_open_times = times - bar_delta.to_timedelta64()
        candle_end_times = times

    # side='right' enforces a strictly later open, even when delay is zero.
    entry_indices = np.searchsorted(
        candle_open_times,
        decision_times + delay_delta.to_timedelta64(),
        side="right",
    )
    entry_available = entry_indices < n
    entry_times = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    entry_times[entry_available] = candle_open_times[entry_indices[entry_available]]

    reasons = {"features_unavailable": int((~feature_ok).sum())}
    reasons["unavailable_entry"] = int((feature_ok & ~entry_available).sum())
    candidate = feature_ok & entry_available
    exit_indices = np.full(n, n, dtype=int)
    if candidate.any():
        targets = entry_times[candidate] + horizon_delta.to_timedelta64()
        exit_indices[candidate] = np.searchsorted(candle_end_times, targets, side="left")
    has_exit = exit_indices < n
    reasons["unavailable_exit"] = int((candidate & ~has_exit).sum())
    candidate &= has_exit

    # Prefix sums check all intervals between decision bar and exit bar in O(1).
    bad_edges = np.zeros(n, dtype=bool)
    if n > 1:
        bad_edges[1:] = np.diff(times) > tolerance.to_timedelta64()
    prefix = np.r_[0, np.cumsum(bad_edges, dtype=np.int64)]
    gap_count = prefix[np.minimum(exit_indices + 1, n)] - prefix[indices + 1]
    gap_free = gap_count == 0
    reasons["gap_exceeds_tolerance"] = int((candidate & ~gap_free).sum())
    candidate &= gap_free

    opens = market["open"].to_numpy(dtype=float)
    closes = market["close"].to_numpy(dtype=float)
    flat = np.zeros(n, dtype=bool)
    flat[candidate] = opens[entry_indices[candidate]] == closes[exit_indices[candidate]]
    reasons["flat_outcome"] = int((candidate & flat).sum())
    accepted = candidate & ~flat
    accepted_indices = indices[accepted]

    samples = featured.iloc[accepted_indices][feature_columns].copy().reset_index(drop=True)
    samples["decision_timestamp"] = pd.to_datetime(decision_times[accepted], utc=True)
    samples["entry_timestamp"] = pd.to_datetime(entry_times[accepted], utc=True)
    samples["exit_timestamp"] = pd.to_datetime(
        candle_end_times[exit_indices[accepted]], utc=True
    )
    samples["entry_price"] = opens[entry_indices[accepted]]
    samples["exit_price"] = closes[exit_indices[accepted]]
    samples["target"] = (samples["exit_price"] > samples["entry_price"]).astype("int8")
    samples["future_outcome_end_time"] = samples["exit_timestamp"]
    if samples.empty:
        samples = pd.DataFrame(columns=feature_columns + SAMPLE_COLUMNS)

    return samples, {
        "market_rows_input": input_rows,
        "market_rows_dropped_invalid_timestamp": dropped_invalid_timestamps,
        "market_duplicate_timestamps_dropped": duplicate_timestamps,
        "market_rows_used": len(market),
        "source_load_report": source_load_report,
        "total_decisions": n,
        "accepted": len(samples),
        "rejected": n - len(samples),
        "rejected_by_reason": reasons,
    }


def chronological_split(
    samples: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    purge_interval: str | pd.Timedelta = "0min",
) -> dict[str, pd.DataFrame]:
    """Split on decision time and purge outcomes that reach the next period.

    Initial boundaries use chronological decision timestamps. Training samples
    are removed when ``future_outcome_end_time + purge_interval`` is at or
    after the validation start; validation samples use the test start likewise.
    The interval adds an embargo after each outcome. Test samples are not
    purged because there is no subsequent split.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    required = {"decision_timestamp", "future_outcome_end_time"}
    if not required.issubset(samples.columns):
        raise ValueError(f"samples must include columns: {sorted(required)}")
    purge = pd.Timedelta(purge_interval)
    if purge < pd.Timedelta(0):
        raise ValueError("purge_interval cannot be negative")

    ordered = samples.copy()
    ordered["decision_timestamp"] = normalize_utc_timestamps(
        ordered["decision_timestamp"]
    )
    ordered["future_outcome_end_time"] = normalize_utc_timestamps(
        ordered["future_outcome_end_time"]
    )
    ordered = ordered.sort_values("decision_timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    validation_start = ordered["decision_timestamp"].iloc[train_end] if train_end < n else pd.NaT
    test_start = ordered["decision_timestamp"].iloc[validation_end] if validation_end < n else pd.NaT
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if not train.empty:
        train = train.loc[train["future_outcome_end_time"] + purge < validation_start]
    if not validation.empty:
        validation = validation.loc[validation["future_outcome_end_time"] + purge < test_start]
    return {
        "train": train.copy(),
        "validation": validation.copy(),
        "test": test,
    }
