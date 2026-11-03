"""Leakage-aware multi-pair feature, label, validation, and candidate training."""
from __future__ import annotations

import argparse
import json
import hashlib
import platform
from importlib.metadata import version as package_version
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.research.data import normalize_utc_timestamps, validate_ohlc_prices
from src.research.features import build_features
from .config import DEFAULT_PAIRS, TIMEFRAME_MINUTES
from .training import expected_calibration_error


BASE_FEATURES = (
    "return_1", "return_5", "return_15", "return_60", "ema_gap", "ema_long_gap",
    "rsi_14", "volatility_15", "volatility_60", "candle_range", "candle_body",
    "time_sin", "time_cos", "macd_pct", "macd_signal_pct", "atr_pct",
    "trend_strength", "volatility_regime", "london_session", "new_york_session",
    "peer_return_mean", "peer_return_dispersion",
)


def canonical_pair(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalpha())[:6]


def _segments(times: pd.Series, tolerance: pd.Timedelta) -> pd.Series:
    gaps = times.diff().gt(tolerance)
    if len(gaps):
        gaps.iloc[0] = False
    return gaps.cumsum()


def build_model_features(candles: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    """Build causal indicator/session/regime features from completed bars only."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    data = candles.copy()
    data["time"] = normalize_utc_timestamps(data["time"])
    if data.time.isna().any() or data.time.duplicated().any():
        raise ValueError("feature candles need valid unique timestamps")
    data = data.sort_values("time", kind="stable").reset_index(drop=True)
    data = validate_ohlc_prices(data)
    data = build_features(data, gap_tolerance=step * 1.5)
    segment = _segments(data.time, step * 1.5)
    close, high, low = data.close.astype(float), data.high.astype(float), data.low.astype(float)
    ema12 = close.groupby(segment, sort=False).transform(
        lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = close.groupby(segment, sort=False).transform(
        lambda x: x.ewm(span=26, adjust=False).mean())
    data["macd_pct"] = (ema12 - ema26) / close
    data["macd_signal_pct"] = data.macd_pct.groupby(segment, sort=False).transform(
        lambda x: x.ewm(span=9, adjust=False).mean())
    prev_close = close.groupby(segment, sort=False).shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(),
                             (low - prev_close).abs()], axis=1).max(axis=1)
    data["atr_pct"] = true_range.groupby(segment, sort=False).transform(
        lambda x: x.rolling(14, min_periods=14).mean()) / close
    data["trend_strength"] = ((data.ema_21 - data.ema_50).abs() / close) / data.atr_pct
    vol_median = data.volatility_15.groupby(segment, sort=False).transform(
        lambda x: x.rolling(100, min_periods=30).median())
    data["volatility_regime"] = data.volatility_15 / vol_median
    hour = data.time.dt.hour
    data["london_session"] = hour.between(7, 15).astype(float)
    data["new_york_session"] = hour.between(12, 20).astype(float)
    return data.replace([np.inf, -np.inf], np.nan)


def build_multi_pair_samples(candles_by_pair: Mapping[str, pd.DataFrame], *, timeframe: str,
                             horizon_bars: int = 3,
                             round_trip_cost_pips: float | Mapping[str, float],
                             pair_universe: Sequence[str] | None = None) -> pd.DataFrame:
    """Create one row per pair/time with exact-bar labels and peer-only features.

    An observation is kept only when entry/exit bars are contiguous at the
    requested timeframe. No candle values are imputed or synthesized.
    """
    if timeframe not in TIMEFRAME_MINUTES or horizon_bars < 1:
        raise ValueError("invalid timeframe or horizon")
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    universe = tuple(sorted({canonical_pair(x) for x in (pair_universe or candles_by_pair)}))
    features_by_pair: dict[str, pd.DataFrame] = {}
    for symbol, raw in candles_by_pair.items():
        pair = canonical_pair(symbol)
        if pair in features_by_pair:
            raise ValueError(f"duplicate canonical pair in training input: {pair}")
        frame = build_model_features(raw, timeframe=timeframe)
        features_by_pair[pair] = frame

    # Cross-pair context uses only completed returns at the same timestamp.
    returns = pd.concat([
        frame[["time", "return_1"]].assign(_pair=pair)
        for pair, frame in features_by_pair.items()
    ], ignore_index=True)
    peer = returns.pivot(index="time", columns="_pair", values="return_1")
    all_rows: list[pd.DataFrame] = []
    for pair, frame in features_by_pair.items():
        peers = [name for name in peer.columns if name != pair]
        if peers:
            peer_values = peer.reindex(frame.time)[peers]
            frame["peer_return_mean"] = peer_values.mean(axis=1, skipna=True).to_numpy()
            frame["peer_return_dispersion"] = peer_values.std(axis=1, skipna=True, ddof=0).to_numpy()
        else:
            frame["peer_return_mean"] = np.nan
            frame["peer_return_dispersion"] = np.nan
        compact = pair
        pip_size = 0.01 if compact.endswith("JPY") else 0.0001
        if isinstance(round_trip_cost_pips, Mapping):
            cost = float(round_trip_cost_pips.get(pair, round_trip_cost_pips.get(next(
                (key for key in round_trip_cost_pips if canonical_pair(key) == pair), ""), np.nan)))
            if not np.isfinite(cost) or cost < 0:
                raise ValueError(f"no valid round-trip cost supplied for {pair}")
        else:
            cost = float(round_trip_cost_pips)
        if cost < 0 or not np.isfinite(cost):
            raise ValueError("round-trip cost must be finite and nonnegative")

        times = pd.DatetimeIndex(frame.time)
        opens, closes = frame.open.to_numpy(float), frame.close.to_numpy(float)
        targets = np.full(len(frame), np.nan)
        future_pips = np.full(len(frame), np.nan)
        future_return = np.full(len(frame), np.nan)
        future_vol_units = np.full(len(frame), np.nan)
        cost_target = np.full(len(frame), np.nan)
        entries = np.full(len(frame), None, dtype=object)
        exits = np.full(len(frame), None, dtype=object)
        # Use disjoint same-pair outcome windows. This prevents the same future
        # candles from serving as labels for neighboring training rows.
        for i in range(0, len(frame) - horizon_bars - 1, horizon_bars + 1):
            entry_i, exit_i = i + 1, i + horizon_bars + 1
            expected = pd.date_range(times[i], periods=horizon_bars + 2, freq=step)
            if not times[i:exit_i + 1].equals(expected):
                continue
            change = closes[exit_i] - opens[entry_i]
            if change == 0:
                continue
            pips = change / pip_size
            targets[i] = int(change > 0)
            future_pips[i] = pips
            future_return[i] = change / opens[entry_i]
            atr_pct = float(frame.atr_pct.iloc[i])
            future_vol_units[i] = future_return[i] / atr_pct if np.isfinite(atr_pct) and atr_pct > 0 else np.nan
            cost_target[i] = int(abs(pips) > cost)
            entries[i], exits[i] = times[entry_i], times[exit_i]

        frame = frame.copy()
        frame["pair"] = pair
        frame["timeframe"] = timeframe
        frame["target"] = targets
        frame["target_exceeds_cost"] = cost_target
        frame["future_pips"] = future_pips
        frame["future_return"] = future_return
        frame["future_volatility_units"] = future_vol_units
        frame["entry_time"] = entries
        frame["label_end_time"] = exits
        frame["round_trip_cost_pips"] = cost
        frame["pip_size"] = pip_size
        for known_pair in universe:
            frame[f"pair_{known_pair}"] = float(pair == known_pair)
        for known_tf in TIMEFRAME_MINUTES:
            frame[f"timeframe_{known_tf}"] = float(timeframe == known_tf)
        required_features = [name for name in BASE_FEATURES if not name.startswith("peer_")]
        all_rows.append(frame.dropna(subset=required_features + ["target", "label_end_time"]))
    if not all_rows:
        return pd.DataFrame()
    result = pd.concat(all_rows, ignore_index=True)
    return result.sort_values(["time", "pair"], kind="stable").reset_index(drop=True)


def _make_model(algorithm: str, seed: int = 42):
    if algorithm == "logistic_regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=1000, random_state=seed))
    if algorithm == "hist_gradient_boosting":
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                                            min_samples_leaf=30,
                                                            l2_regularization=1.0,
                                                            random_state=seed))
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _trade_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float = 0.55) -> dict:
    ranked = frame.reset_index(drop=True).assign(_probability=probability).sort_values(
        ["entry_time", "pair"], kind="stable")
    selected = []
    exit_by_pair: dict[str, pd.Timestamp] = {}
    for _, row in ranked.iterrows():
        p = row["_probability"]
        direction = 1 if p >= threshold else -1 if p <= 1 - threshold else 0
        if direction == 0:
            continue
        pair = row.pair
        if pair in exit_by_pair and row.entry_time < exit_by_pair[pair]:
            continue
        net_pips = direction * row.future_pips - row.round_trip_cost_pips
        selected.append((row.entry_time, pair, float(net_pips),
                         float(direction * row.future_return - row.round_trip_cost_pips * row.pip_size / row.close),
                         direction, int(row.target)))
        exit_by_pair[pair] = row.label_end_time
    if not selected:
        return {"trades": 0, "net_expectancy_pips": 0.0, "profit_factor": 0.0,
                "net_return": 0.0,
                "max_drawdown_return": 0.0, "precision": 0.0, "recall": 0.0,
                "directional_accuracy": 0.0}
    net = np.asarray([item[2] for item in selected])
    returns = np.asarray([item[3] for item in selected])
    equity = np.cumsum(returns)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]
    wins, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
    buys = [item for item in selected if item[4] > 0]
    true_positive_buys = sum(item[5] == 1 for item in buys)
    all_positive = int(frame.target.sum())
    return {"trades": len(selected), "net_expectancy_pips": float(net.mean()),
            "net_return": float(returns.sum()),
            "profit_factor": float(wins / losses) if losses else (None if wins else 0.0),
            "max_drawdown_return": float(drawdown.min()),
            "precision": float(true_positive_buys / len(buys)) if buys else 0.0,
            "recall": float(true_positive_buys / all_positive) if all_positive else 0.0,
            "directional_accuracy": float(np.mean([
                (item[4] > 0 and item[5] == 1) or (item[4] < 0 and item[5] == 0)
                for item in selected]))}


def _metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    actual = frame.target.astype(int).to_numpy()
    auc = float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None
    trades = _trade_metrics(frame, probability)
    return {"rows": int(len(frame)), "auc": auc,
            "brier_score": float(brier_score_loss(actual, probability)),
            "baseline_brier_score": float(brier_score_loss(actual, np.full(len(actual), actual.mean()))),
            "expected_calibration_error": expected_calibration_error(actual, probability),
            **trades}


def _walk_forward(samples: pd.DataFrame, columns: Sequence[str], *, folds: int = 3,
                  minimum_train_rows: int = 300,
                  minimum_validation_rows: int = 30) -> dict[str, list[dict]]:
    times = pd.Index(samples.time.drop_duplicates().sort_values())
    if len(times) < 20:
        return {algorithm: [] for algorithm in ("logistic_regression", "hist_gradient_boosting")}
    cut_points = np.linspace(0.50, 0.80, folds + 1)
    results = {algorithm: [] for algorithm in ("logistic_regression", "hist_gradient_boosting")}
    for left, right in zip(cut_points[:-1], cut_points[1:]):
        start_i, stop_i = int(len(times) * left), int(len(times) * right)
        if start_i >= len(times) or stop_i <= start_i:
            continue
        validation_start = times[start_i]
        validation_end = times[min(stop_i - 1, len(times) - 1)]
        train = samples.loc[samples.label_end_time < validation_start]
        validation = samples.loc[(samples.time >= validation_start) & (samples.time <= validation_end)]
        if (len(train) < minimum_train_rows or len(validation) < minimum_validation_rows
                or validation.target.nunique() < 2):
            continue
        for algorithm in results:
            model = _make_model(algorithm)
            model.fit(train[list(columns)], train.target.astype(int))
            prob = model.predict_proba(validation[list(columns)])[:, 1]
            results[algorithm].append(_metrics(validation, prob))
    return results


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["time", "pair", *BASE_FEATURES, "target", "future_pips",
               "round_trip_cost_pips", "label_end_time"]
    hashed = pd.util.hash_pandas_object(
        frame[columns].sort_values(["time", "pair"], kind="stable"),
        index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def train_multi_pair_candidates(candles_by_pair: Mapping[str, pd.DataFrame], *, timeframe: str,
                                output_dir: str | Path,
                                round_trip_cost_pips: float | Mapping[str, float],
                                horizon_bars: int = 3,
                                minimum_train_rows: int = 500,
                                minimum_holdout_rows: int = 100,
                                minimum_trades: int = 50,
                                minimum_walk_forward_validation_rows: int = 30) -> list[dict]:
    """Train shared and per-pair logistic/tree candidates; never promotes them."""
    samples = build_multi_pair_samples(
        candles_by_pair, timeframe=timeframe, horizon_bars=horizon_bars,
        round_trip_cost_pips=round_trip_cost_pips,
        pair_universe=tuple(candles_by_pair),
    )
    if samples.empty:
        raise ValueError("no contiguous labeled samples were available")
    output_dir = Path(output_dir)
    universe = sorted(samples.pair.unique())
    groups: list[tuple[str, str | None, pd.DataFrame]] = [("shared", None, samples)]
    groups.extend(("pair", pair, samples.loc[samples.pair == pair].reset_index(drop=True))
                  for pair in universe)
    reports: list[dict] = []
    for kind, pair, group in groups:
        group = group.sort_values(["time", "pair"], kind="stable").reset_index(drop=True)
        unique_times = group.time.drop_duplicates().sort_values()
        if len(group) < minimum_train_rows + minimum_holdout_rows or len(unique_times) < 50:
            reports.append({"kind": kind, "symbol": pair, "timeframe": timeframe,
                            "status": "insufficient_data", "rows": int(len(group))})
            continue
        holdout_start = unique_times.iloc[int(len(unique_times) * .85)]
        train = group.loc[group.label_end_time < holdout_start].copy()
        holdout = group.loc[group.time >= holdout_start].copy()
        if len(train) < minimum_train_rows or len(holdout) < minimum_holdout_rows:
            reports.append({"kind": kind, "symbol": pair, "timeframe": timeframe,
                            "status": "insufficient_purged_split", "train_rows": len(train),
                            "holdout_rows": len(holdout)})
            continue
        columns = [*BASE_FEATURES,
                   *(f"pair_{known}" for known in universe),
                   *(f"timeframe_{known}" for known in TIMEFRAME_MINUTES)]
        wf = _walk_forward(group.loc[group.time < holdout_start].copy(), columns,
                           minimum_train_rows=max(50, minimum_train_rows // 2),
                           minimum_validation_rows=minimum_walk_forward_validation_rows)
        for algorithm, folds in wf.items():
            fold_brier = [fold["brier_score"] for fold in folds]
            cv_summary = {
                "folds": len(folds),
                "mean_auc": float(np.mean([fold["auc"] for fold in folds])) if folds else None,
                "mean_brier_score": float(np.mean(fold_brier)) if folds else None,
                "mean_baseline_brier_score": float(np.mean(
                    [fold["baseline_brier_score"] for fold in folds])) if folds else None,
                "mean_expected_calibration_error": float(np.mean(
                    [fold["expected_calibration_error"] for fold in folds])) if folds else None,
                "mean_net_expectancy_pips": float(np.mean(
                    [fold["net_expectancy_pips"] for fold in folds])) if folds else None,
            }
            model = _make_model(algorithm)
            model.fit(train[columns], train.target.astype(int))
            holdout_probability = model.predict_proba(holdout[columns])[:, 1]
            holdout_metrics = _metrics(holdout, holdout_probability)
            folds_pass = len(folds) >= 2
            cv_pass = (folds_pass and cv_summary["mean_auc"] is not None
                       and cv_summary["mean_auc"] > .5
                       and cv_summary["mean_brier_score"] < cv_summary["mean_baseline_brier_score"]
                       and cv_summary["mean_expected_calibration_error"] <= .05
                       and cv_summary["mean_net_expectancy_pips"] > 0)
            gate = (cv_pass and holdout_metrics["auc"] is not None
                    and holdout_metrics["auc"] > .5
                    and holdout_metrics["brier_score"] < holdout_metrics["baseline_brier_score"]
                    and holdout_metrics["expected_calibration_error"] <= .05
                    and holdout_metrics["trades"] >= minimum_trades
                    and holdout_metrics["net_expectancy_pips"] > 0)
            trained_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            data_hash = _dataset_fingerprint(group)
            model_id = f"{kind}_{pair or 'ALL'}_{timeframe}_{algorithm}"
            metadata = {
                "version": trained_at, "model_id": model_id, "model_kind": kind,
                "symbol": pair, "timeframe": timeframe, "horizon_bars": horizon_bars,
                "bar_minutes": TIMEFRAME_MINUTES[timeframe],
                "algorithm": algorithm, "feature_columns": columns,
                "feature_pipeline_version": "multi_pair_v1",
                "runtime_versions": {"python": platform.python_version(),
                                     "pandas": pd.__version__,
                                     "numpy": np.__version__,
                                     "scikit_learn": package_version("scikit-learn")},
                "random_seed": 42,
                "pair_universe": universe, "training_rows": int(len(train)),
                "training_prevalence": float(train.target.mean()),
                "walk_forward_folds": folds,
                "walk_forward_summary": cv_summary,
                "walk_forward_mean_brier": float(np.mean(fold_brier)) if fold_brier else None,
                "holdout_start_utc": holdout_start.isoformat(),
                "holdout_end_utc": holdout.label_end_time.max().isoformat(),
                "training_end_utc": train.time.max().isoformat(),
                "dataset_fingerprint": data_hash,
                "source_provenance": (group[["source", "collected_at_utc"]]
                                      .drop_duplicates().to_dict("records")
                                      if {"source", "collected_at_utc"}.issubset(group.columns)
                                      else []),
                "label": "direction of next-open to horizon-close return",
                "horizon_bars": horizon_bars,
                "label_policy": {"outcome_windows_non_overlapping": True,
                                 "skip_missing_intervals": True,
                                 "cost_target": "absolute future pips exceeds configured round-trip pips"},
                "validation_policy": {"walk_forward": "expanding chronological folds",
                                      "purge": "training label end must precede validation start",
                                      "final_holdout_fraction": 0.15},
                "cost_assumption_round_trip_pips": group.round_trip_cost_pips.unique().tolist(),
                "trade_simulation_assumptions": (
                    "entry at next bar open, exit at configured horizon close; "
                    "fixed threshold 0.55/0.45; no overlapping positions per pair; "
                    "equal notional return aggregation; all spread/commission/slippage "
                    "represented by the explicit round-trip pip cost; financing and "
                    "market impact are not modeled"),
                "holdout_metrics": holdout_metrics, "gate_passed": bool(gate),
                "status": "candidate_only",
                "final_holdout_usage": "reported once during candidate evaluation; exclude from later confirmation",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{model_id}_{trained_at}.joblib"
            artifact = output_dir / filename
            joblib.dump({"status": "candidate", "model": model,
                         "feature_columns": columns, "metadata": metadata}, artifact)
            report = {"kind": kind, "symbol": pair, "timeframe": timeframe,
                      "algorithm": algorithm, "candidate_path": str(artifact),
                      "status": "candidate_created", "gate_passed": bool(gate),
                      "walk_forward_folds": len(folds), "holdout": holdout_metrics}
            reports.append(report)
            _register_candidate(output_dir.parent / "candidate_registry.json", report, metadata)
    return reports


def _register_candidate(path: Path, report: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"candidates": []}
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"candidate registry is corrupt: {path}")
    current.setdefault("candidates", []).append({**report, "metadata": metadata})
    path.write_text(json.dumps(current, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_collected_frames(data_dir: str | Path, pairs: Sequence[str], timeframe: str) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    result = {}
    for pair in pairs:
        canonical = canonical_pair(pair)
        path = root / f"{canonical}_{timeframe}.parquet"
        if path.exists():
            if timeframe == "H4" and canonical in {"EURUSD", "USDJPY"}:
                from src.research.multi_strategy_engine import load_research_frame

                if root.name == "mt5-history":
                    derived_dir, quarantine_dir = root.parent / "derived-research", root.parent / "quarantine"
                else:
                    derived_dir, quarantine_dir = root / "derived-research", root / "quarantine"
                frame, _ = load_research_frame(
                    canonical, timeframe, root,
                    derived_dir=derived_dir, quarantine_dir=quarantine_dir)
            else:
                frame = pd.read_parquet(path)
                frame["time"] = pd.to_datetime(frame.time, utc=True)
            result[pair] = frame
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inert multi-pair MT5 model candidates")
    parser.add_argument("--data-dir", default="market-data/mt5-history")
    parser.add_argument("--output-dir", default="models/candidates")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--timeframes", default="M5,M15,H1,H4")
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--round-trip-cost-pips", type=float, required=True,
                        help="explicit assumed total spread+commission+slippage cost")
    args = parser.parse_args()
    pairs = [x.strip().upper() for x in args.pairs.split(",") if x.strip()]
    all_reports = []
    for timeframe in [x.strip().upper() for x in args.timeframes.split(",") if x.strip()]:
        frames = load_collected_frames(args.data_dir, pairs, timeframe)
        if not frames:
            all_reports.append({"timeframe": timeframe, "status": "no_collected_data"})
            continue
        all_reports.extend(train_multi_pair_candidates(
            frames, timeframe=timeframe, output_dir=args.output_dir,
            round_trip_cost_pips=args.round_trip_cost_pips,
            horizon_bars=args.horizon_bars,
        ))
    for report in all_reports:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    if not all_reports or all(item.get("status") == "no_collected_data" for item in all_reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
