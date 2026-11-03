from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.research.features import build_features
from src.research.data import normalize_utc_timestamps, validate_ohlc_prices

FEATURES = ("return_1", "return_5", "return_15", "return_60", "ema_gap",
            "ema_long_gap", "rsi_14", "volatility_15", "volatility_60",
            "candle_range", "candle_body", "time_sin", "time_cos")


@dataclass(frozen=True)
class CandidateResult:
    candidate_path: str
    train_rows: int
    validation_rows: int
    auc: float
    brier_score: float
    baseline_brier_score: float
    net_expectancy_after_assumed_costs: float
    validation_trades: int
    gate_passed: bool
    promotion_status: str = "candidate_only"


def make_samples(candles: pd.DataFrame, *, horizon_minutes: int = 15,
                 pip_size: float = 0.0001, round_trip_cost_pips: float = 1.0) -> pd.DataFrame:
    """Use the existing causal feature builder and exact elapsed-time outcomes."""
    if horizon_minutes <= 0 or pip_size <= 0 or round_trip_cost_pips < 0:
        raise ValueError("invalid labeling parameters")
    data = candles.copy()
    data["time"] = normalize_utc_timestamps(data["time"])
    if data["time"].isna().any() or data.time.duplicated().any() or len(data) < 300:
        raise ValueError("at least 300 valid, unique candles are required")
    data = data.sort_values("time").reset_index(drop=True)
    data = validate_ohlc_prices(data)
    featured = build_features(data, gap_tolerance=pd.Timedelta(seconds=90))
    times = pd.DatetimeIndex(featured.time)
    targets = np.full(len(featured), np.nan)
    returns = np.full(len(featured), np.nan)
    exit_indices = np.full(len(featured), -1, dtype=int)
    entry_times = np.full(len(featured), None, dtype=object)
    exit_times = np.full(len(featured), None, dtype=object)
    opens = featured.open.to_numpy(float)
    closes = featured.close.to_numpy(float)
    lookup = {stamp: i for i, stamp in enumerate(times)}
    for i, stamp in enumerate(times):
        entry_i = lookup.get(stamp + pd.Timedelta(minutes=1))
        if entry_i is None:
            continue
        exit_i = lookup.get(stamp + pd.Timedelta(minutes=horizon_minutes + 1))
        if exit_i is None or exit_i <= entry_i:
            continue
        if (times[entry_i:exit_i + 1].to_series().diff().dropna() > pd.Timedelta("90s")).any():
            continue
        move = (closes[exit_i] - opens[entry_i]) / pip_size
        if move == 0:
            continue
        targets[i], returns[i], exit_indices[i] = int(move > 0), move, exit_i
        entry_times[i], exit_times[i] = times[entry_i], times[exit_i]
    featured["target"] = targets
    featured["future_pips"] = returns
    featured["exit_index"] = exit_indices
    featured["entry_time"] = entry_times
    featured["exit_time"] = exit_times
    return featured.dropna(subset=list(FEATURES) + ["target"]).reset_index(drop=True)


def _score_non_overlapping(part: pd.DataFrame, probability: np.ndarray,
                           threshold: float, cost: float) -> tuple[int, float]:
    selected: list[float] = []
    allowed_exit = None
    for row, p in zip(part.itertuples(index=False), probability):
        if p >= threshold:
            direction = 1
        elif p <= 1 - threshold:
            direction = -1
        else:
            continue
        exit_index = int(getattr(row, "exit_index"))
        entry_time = getattr(row, "entry_time")
        exit_time = getattr(row, "exit_time")
        if (allowed_exit is not None and entry_time < allowed_exit) or exit_index < 0:
            continue
        selected.append(direction * float(getattr(row, "future_pips")) - cost)
        allowed_exit = exit_time
    return len(selected), float(np.mean(selected)) if selected else 0.0


def expected_calibration_error(actual: np.ndarray, probability: np.ndarray,
                               bins: int = 10) -> float:
    """Sample-weighted calibration error over fixed equal-width bins."""
    y, p = np.asarray(actual, dtype=float), np.asarray(probability, dtype=float)
    if len(y) != len(p) or not len(y) or not 2 <= bins <= 50:
        raise ValueError("invalid calibration inputs")
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean() - p[mask].mean()))
    return float(error)


def train_candidate(candles: pd.DataFrame, output_path: str | Path, *,
                    horizon_minutes: int = 15, assumed_cost_pips: float = 1.0,
                    minimum_validation_samples: int = 500,
                    minimum_validation_trades: int = 100) -> CandidateResult:
    ordered = candles.copy()
    ordered["time"] = normalize_utc_timestamps(ordered["time"])
    if ordered.time.isna().any() or ordered.time.duplicated().any():
        raise ValueError("training candles must have valid unique timestamps")
    ordered = ordered.sort_values("time").reset_index(drop=True)
    # Reserve the last 20% of source candles completely: do not build labels,
    # features, or metrics from that final chronological interval.
    development = ordered.iloc[:int(len(ordered) * .8)].copy()
    samples = make_samples(development, horizon_minutes=horizon_minutes,
                           round_trip_cost_pips=assumed_cost_pips)
    n = len(samples)
    train_end = int(n * .75)
    # Purge the outcome horizon at each chronological split boundary.
    train = samples.iloc[:max(0, train_end - horizon_minutes - 1)]
    validation = samples.iloc[train_end + horizon_minutes + 1:]
    if len(train) < 300 or len(validation) < minimum_validation_samples:
        raise ValueError("insufficient purged chronological training/validation samples")
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[list(FEATURES)], train.target.astype(int))
    probabilities = model.predict_proba(validation[list(FEATURES)])[:, 1]
    y = validation.target.astype(int).to_numpy()
    auc = float(roc_auc_score(y, probabilities))
    brier = float(brier_score_loss(y, probabilities))
    baseline_brier = float(brier_score_loss(y, np.full(len(y), train.target.mean())))
    calibration_error = expected_calibration_error(y, probabilities)
    # Candidate threshold is fixed before validation; overlapping positions are excluded.
    trades, expectancy = _score_non_overlapping(validation, probabilities, .55, assumed_cost_pips)
    gate = (len(validation) >= minimum_validation_samples and trades >= minimum_validation_trades
            and auc > .5 and brier < baseline_brier and calibration_error <= .05 and expectancy > 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .model_registry import _data_fingerprint
    data_hash = _data_fingerprint(development)
    bundle = {"status": "candidate", "model": model, "feature_columns": list(FEATURES),
              "metadata": {"algorithm": "logistic_regression", "horizon_minutes": horizon_minutes,
                           "training_rows": len(train), "validation_rows": len(validation),
                           "training_end_utc": train.time.max().isoformat(),
                           "development_end_utc": development.time.max().isoformat(),
                           "timeframe": "M1", "training_prevalence": float(train.target.mean()),
                           "data_fingerprint": data_hash, "assumed_round_trip_cost_pips": assumed_cost_pips,
                           "threshold": .55, "auc": auc, "brier_score": brier,
                           "expected_calibration_error": calibration_error,
                           "baseline_brier_score": baseline_brier,
                           "net_expectancy_after_assumed_costs": expectancy,
                           "validation_trades": trades, "gate_passed": gate,
                           "final_20_percent": "not read for features, labels, or metrics",
                           "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}}
    joblib.dump(bundle, output_path)
    return CandidateResult(str(output_path), len(train), len(validation), auc, brier,
                           baseline_brier, expectancy, trades, gate)


def train_candidate_from_labeled_samples(labeled_samples: list[dict[str, Any]],
                                         output_path: str | Path, *, symbol: str,
                                         timeframe: str = "M1",
                                         assumed_cost_pips: float = 1.0,
                                         minimum_validation_samples: int = 100,
                                         minimum_validation_trades: int = 20) -> CandidateResult:
    """Create an inert candidate from previously logged, explicitly labeled predictions.

    This path intentionally supports M1 predictions only because the observed
    outcome contract is a 15-minute directional return.
    """
    if timeframe != "M1":
        raise ValueError("logged-outcome retraining currently supports M1 only")
    data = pd.DataFrame(labeled_samples)
    required = set(FEATURES) | {"time", "target", "future_pips"}
    if not required.issubset(data.columns):
        raise ValueError(f"labeled observations are missing columns: {sorted(required - set(data.columns))}")
    data["time"] = normalize_utc_timestamps(data["time"])
    if data.time.isna().any() or data.time.duplicated().any():
        raise ValueError("labeled observations must have unique valid candle times")
    data = data.sort_values("time").reset_index(drop=True)
    data["target"] = pd.to_numeric(data.target, errors="coerce")
    data["future_pips"] = pd.to_numeric(data.future_pips, errors="coerce")
    if data.target.isna().any() or not data.target.isin([0, 1]).all() or data.future_pips.isna().any():
        raise ValueError("labeled outcomes must have binary targets and numeric returns")
    n = len(data)
    dev_end = int(n * .8)
    development = data.iloc[:dev_end].copy()
    train_end = int(len(development) * .75)
    train = development.iloc[:max(0, train_end - 16)].copy()
    validation = development.iloc[train_end + 16:].copy()
    if len(train) < 300 or len(validation) < minimum_validation_samples:
        raise ValueError("insufficient newly labeled observations after chronological purge")
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[list(FEATURES)], train.target.astype(int))
    probability = model.predict_proba(validation[list(FEATURES)])[:, 1]
    y = validation.target.astype(int).to_numpy()
    auc = float(roc_auc_score(y, probability))
    brier = float(brier_score_loss(y, probability))
    baseline = float(brier_score_loss(y, np.full(len(y), train.target.mean())))
    calibration_error = expected_calibration_error(y, probability)
    validation = validation.copy()
    validation["entry_time"] = validation.time + pd.Timedelta(minutes=1)
    validation["exit_time"] = validation.time + pd.Timedelta(minutes=16)
    validation["exit_index"] = np.arange(len(validation))
    trades, expectancy = _score_non_overlapping(validation, probability, .55, assumed_cost_pips)
    gate = (trades >= minimum_validation_trades and auc > .5 and brier < baseline
            and calibration_error <= .05 and expectancy > 0)
    from .model_registry import _data_fingerprint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"status": "candidate", "model": model, "feature_columns": list(FEATURES),
              "metadata": {"algorithm": "logistic_regression", "horizon_minutes": 15,
                           "timeframe": timeframe, "symbol": symbol,
                           "training_rows": len(train), "validation_rows": len(validation),
                           "training_end_utc": train.time.max().isoformat(),
                           "development_end_utc": development.time.max().isoformat(),
                           "training_prevalence": float(train.target.mean()),
                           "data_fingerprint": _data_fingerprint(development),
                           "assumed_round_trip_cost_pips": assumed_cost_pips,
                           "threshold": .55, "auc": auc, "brier_score": brier,
                           "baseline_brier_score": baseline,
                           "expected_calibration_error": calibration_error,
                           "net_expectancy_after_assumed_costs": expectancy,
                           "validation_trades": trades, "gate_passed": gate,
                           "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}}
    joblib.dump(bundle, output_path)
    return CandidateResult(str(output_path), len(train), len(validation), auc, brier,
                           baseline, expectancy, trades, gate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an inert Forex model candidate; never auto-promotes")
    parser.add_argument("--data", default="eurusd_clean.csv")
    parser.add_argument("--output", default="models/candidates/candidate.joblib")
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    result = train_candidate(data, args.output)
    print(asdict(result))
    print("Candidate is not active. Independent confirmation and explicit promotion are required.")


if __name__ == "__main__":
    main()
