"""Frozen, training-only predictability map and dependence-aware inference."""

from __future__ import annotations

from math import erfc, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .experiment_store import code_provenance
from .labels import MID_RETURN_FEATURE_COLUMNS, MID_RETURN_HORIZONS


MAP_FAMILY_ID = "eurusd_m1_causal_predictability_v1"
FAMILYWISE_ALPHA = 0.05
HAC_LAG_MINUTES = 120
SHIFT_MINUTES = 120
SHIFT_RESAMPLES = 999
SHIFT_RANDOM_SEED = 20261027
NORMAL_95 = 1.959963984540054
MIN_SHIFT_SEGMENT_ROWS = 2 * SHIFT_MINUTES + 1


def _contiguous_segments(times: pd.Series) -> list[np.ndarray]:
    differences = times.diff().dt.total_seconds().to_numpy()
    boundaries = np.flatnonzero(differences[1:] != 60.0) + 1
    return [
        segment for segment in np.split(np.arange(len(times)), boundaries)
        if len(segment)
    ]


def _hac_se(influence: np.ndarray, segments: list[np.ndarray]) -> float:
    """Newey-West standard error with a frozen 120-observed-minute lag."""
    n = len(influence)
    long_run_variance = 0.0
    for indices in segments:
        values = influence[indices]
        segment_n = len(values)
        if segment_n == 0:
            continue
        max_lag = min(HAC_LAG_MINUTES, segment_n - 1)
        if segment_n >= 2048:
            fft_length = 1 << (2 * segment_n - 1).bit_length()
            spectrum = np.fft.rfft(values, n=fft_length)
            products = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_length)
            autocovariance_sums = products[:max_lag + 1]
        else:
            autocovariance_sums = np.array([
                np.dot(values[:segment_n - lag], values[lag:])
                for lag in range(max_lag + 1)
            ])
        long_run_variance += autocovariance_sums[0]
        if max_lag:
            lags = np.arange(1, max_lag + 1, dtype=float)
            weights = 1.0 - lags / (HAC_LAG_MINUTES + 1.0)
            long_run_variance += 2.0 * float(np.dot(weights, autocovariance_sums[1:]))
    variance = max(0.0, long_run_variance / (n * n))
    return sqrt(variance)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str]:
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None, "nonfinite_input"
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_scale = float(np.sqrt(np.mean(left_centered**2)))
    right_scale = float(np.sqrt(np.mean(right_centered**2)))
    if left_scale == 0.0 or right_scale == 0.0:
        return None, "constant_series"
    value = float(np.mean(left_centered * right_centered) / (left_scale * right_scale))
    return float(np.clip(value, -1.0, 1.0)), "ok"


def evaluate_predictability_map(
    samples: pd.DataFrame,
    *,
    training_boundary_utc: str,
) -> dict[str, Any]:
    """Evaluate the predeclared 16-feature by four-horizon training family.

    The primary effect is Pearson correlation. Inference uses a Newey-West
    HAC standard error (120-minute Bartlett lag), two-sided normal-reference
    p-values, and Holm family-wise adjustment over the fixed 64 tests. A
    seeded circular-shift max-statistic control shifts entire feature series
    within contiguous M1 runs by at least 120 minutes and reports
    family-wise empirical p-values.
    """
    boundary = pd.Timestamp(training_boundary_utc)
    if boundary.tzinfo is None:
        raise ValueError("training_boundary_utc must be timezone-aware")
    boundary = boundary.tz_convert("UTC")
    required = {
        "decision_timestamp", "entry_timestamp", "exit_timestamp", "horizon",
        "forward_log_return", "accepted", *MID_RETURN_FEATURE_COLUMNS,
    }
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"predictability samples are missing columns: {sorted(missing)}")
    if samples.empty:
        raise ValueError("predictability map requires nonempty training samples")
    if not pd.api.types.is_bool_dtype(samples["accepted"].dtype):
        raise ValueError("accepted must be a boolean column")
    if not samples["horizon"].isin(MID_RETURN_HORIZONS).all():
        raise ValueError("samples contain horizons outside the frozen map family")

    accepted = samples.loc[samples["accepted"].astype(bool)].copy()
    if accepted.empty:
        raise ValueError("predictability map has no accepted training labels")
    for column in ("decision_timestamp", "entry_timestamp", "exit_timestamp"):
        accepted[column] = pd.to_datetime(accepted[column], utc=True, errors="raise")
        if (accepted[column] >= boundary).any():
            raise ValueError(
                f"accepted {column} must be strictly before the frozen training boundary"
            )
    if not np.isfinite(accepted["forward_log_return"].to_numpy(dtype=float)).all():
        raise ValueError("accepted labels must have finite forward log returns")

    # Use common complete cases and decisions across horizons so the frozen
    # family and joint circular-shift null retain cross-horizon dependence.
    complete_mask = accepted[list(MID_RETURN_FEATURE_COLUMNS)].notna().all(axis=1)
    complete = accepted.loc[complete_mask].copy()
    if complete.empty:
        raise ValueError("no accepted labels have complete predeclared features")

    by_horizon: dict[str, pd.DataFrame] = {}
    decision_sets: list[set[pd.Timestamp]] = []
    for horizon in MID_RETURN_HORIZONS:
        subset = complete.loc[complete["horizon"] == horizon].sort_values(
            "decision_timestamp", kind="stable",
        )
        if subset["decision_timestamp"].duplicated().any():
            raise ValueError(f"duplicate decision timestamps for horizon {horizon}")
        if subset.empty:
            raise ValueError(f"no complete accepted training labels for horizon {horizon}")
        by_horizon[horizon] = subset
        decision_sets.append(set(subset["decision_timestamp"]))
    common_decisions = set.intersection(*decision_sets)
    if not common_decisions:
        raise ValueError("no common accepted decision timestamps across all horizons")
    common_times = pd.DatetimeIndex(sorted(common_decisions))
    common_segments = _contiguous_segments(pd.Series(common_times))
    eligible_segments = [
        group for group in common_segments if len(group) >= MIN_SHIFT_SEGMENT_ROWS
    ]
    excluded_short_rows = sum(
        len(group) for group in common_segments
        if len(group) < MIN_SHIFT_SEGMENT_ROWS
    )
    if not eligible_segments:
        raise ValueError(
            f"no common contiguous decision segment is long enough for the "
            f"{SHIFT_MINUTES}-minute shift null"
        )
    selected_positions = np.concatenate(eligible_segments)
    analysis_times = common_times.take(selected_positions)
    analysis_decisions = set(analysis_times)

    per_test: list[dict[str, Any]] = []
    primary_p_values: list[float] = []
    test_arrays: dict[str, tuple[np.ndarray, np.ndarray, list[np.ndarray], float | None]] = {}
    horizon_audit: dict[str, Any] = {}
    for horizon in MID_RETURN_HORIZONS:
        subset = by_horizon[horizon].loc[
            by_horizon[horizon]["decision_timestamp"].isin(analysis_decisions)
        ].sort_values("decision_timestamp", kind="stable").reset_index(drop=True)
        segments = _contiguous_segments(subset["decision_timestamp"])
        y = subset["forward_log_return"].to_numpy(dtype=float)
        y_centered = y - y.mean()
        y_scale = float(np.sqrt(np.mean(y_centered**2)))
        if y_scale == 0.0:
            raise ValueError(f"all forward returns are constant for horizon {horizon}")

        horizon_audit[horizon] = {
            "accepted_complete_rows": int(len(by_horizon[horizon])),
            "horizon_intersection_excluded": int(
                len(by_horizon[horizon]) - len(common_decisions)
            ),
            "rows_used": int(len(subset)),
            "short_segments_excluded": int(excluded_short_rows),
            "contiguous_segments": int(len(segments)),
            "shift_eligible_segments": int(len(eligible_segments)),
        }
        for feature in MID_RETURN_FEATURE_COLUMNS:
            x = subset[feature].to_numpy(dtype=float)
            correlation, status = _correlation(x, y)
            if correlation is None:
                standard_error = None
                p_value = 1.0
                lower = upper = None
            else:
                x_centered = x - x.mean()
                x_scale = float(np.sqrt(np.mean(x_centered**2)))
                x_standard = x_centered / x_scale
                y_standard = y_centered / y_scale
                influence = (
                    x_standard * y_standard
                    - correlation * (x_standard**2 + y_standard**2) / 2.0
                )
                standard_error = _hac_se(influence, segments)
                if standard_error == 0.0:
                    p_value = 0.0 if correlation != 0.0 else 1.0
                else:
                    p_value = erfc(abs(correlation / standard_error) / sqrt(2.0))
                lower = max(-1.0, correlation - NORMAL_95 * standard_error)
                upper = min(1.0, correlation + NORMAL_95 * standard_error)
            primary_p_values.append(float(p_value))
            per_test.append({
                "feature": feature,
                "horizon": horizon,
                "n": int(len(subset)),
                "correlation": correlation,
                "hac_standard_error": standard_error,
                "hac_95_ci": [lower, upper] if lower is not None else None,
                "hac_p_value": float(p_value),
                "status": status,
            })
            test_arrays[f"{feature}|{horizon}"] = (
                x, y, segments, correlation,
            )

    adjusted = _holm_adjust(np.asarray(primary_p_values, dtype=float))
    for result, adjusted_p in zip(per_test, adjusted):
        result["holm_adjusted_p_value"] = float(adjusted_p)
        result["significant_holm_0_05"] = bool(adjusted_p <= FAMILYWISE_ALPHA)

    # Share shifts across all features and horizons to preserve their joint
    # temporal dependence in each max-statistic null draw.
    rng = np.random.default_rng(SHIFT_RANDOM_SEED)
    null_correlations = np.zeros((SHIFT_RESAMPLES, len(per_test)), dtype=float)
    offsets_by_segment = [
        rng.integers(
            SHIFT_MINUTES, len(group) - SHIFT_MINUTES + 1,
            size=SHIFT_RESAMPLES,
        )
        for group in _contiguous_segments(pd.Series(analysis_times))
    ]
    for test_index, result in enumerate(per_test):
        correlation = result["correlation"]
        if correlation is None:
            continue
        x, y, groups, _ = test_arrays[f"{result['feature']}|{result['horizon']}"]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        x_scale = float(np.sqrt(np.mean(x_centered**2)))
        y_scale = float(np.sqrt(np.mean(y_centered**2)))
        denominator = len(x) * x_scale * y_scale
        for group, offsets in zip(groups, offsets_by_segment):
            group_x = x_centered[group]
            group_y = y_centered[group]
            group_n = len(group)
            x_fft = np.fft.rfft(group_x, n=group_n)
            y_fft = np.fft.rfft(group_y, n=group_n)
            cross_products = np.fft.irfft(
                np.conjugate(x_fft) * y_fft, n=group_n,
            )
            null_correlations[:, test_index] += cross_products[offsets] / denominator

    max_null = np.max(np.abs(null_correlations), axis=1)
    for result in per_test:
        if result["correlation"] is None:
            shift_p = 1.0
        else:
            shift_p = (
                1 + int(np.count_nonzero(max_null >= abs(result["correlation"])))
            ) / (SHIFT_RESAMPLES + 1)
        result["shift_max_statistic_fwer_p_value"] = float(shift_p)

    return {
        "family_id": MAP_FAMILY_ID,
        "family": {
            "features": list(MID_RETURN_FEATURE_COLUMNS),
            "horizons": list(MID_RETURN_HORIZONS),
            "size": len(MID_RETURN_FEATURE_COLUMNS) * len(MID_RETURN_HORIZONS),
            "feature_horizon_tests": len(per_test),
        },
        "training_boundary_utc": boundary.isoformat(),
        "inference": {
            "effect": "Pearson correlation",
            "uncertainty": "Newey-West HAC influence-function standard error, Bartlett kernel, 120 one-minute lags; contiguous M1 segments are not joined",
            "primary_p_value": "two-sided normal-reference HAC Wald p-value",
            "multiple_testing": "Holm step-down family-wise correction over all 64 frozen feature-horizon tests at alpha=0.05",
            "null_control": "999 seeded circular feature shifts within contiguous M1 runs; each offset is at least 120 minutes; max-absolute-correlation controls family-wise error over all 64 tests",
            "shift_seed": SHIFT_RANDOM_SEED,
            "shift_resamples": SHIFT_RESAMPLES,
            "minimum_shift_segment_rows": MIN_SHIFT_SEGMENT_ROWS,
        },
        "sample_accounting": {
            "accepted_label_rows": int(len(accepted)),
            "complete_feature_rows": int(len(complete)),
            "dropped_incomplete_feature_rows": int(len(accepted) - len(complete)),
            "common_decision_timestamps": int(len(common_decisions)),
            "dropped_common_segments_below_shift_minimum": int(excluded_short_rows),
            "analysis_decision_timestamps": int(len(analysis_times)),
            "by_horizon": horizon_audit,
        },
        "software_provenance": code_provenance(),
        "results": per_test,
    }
