"""Mission 12 leakage-aware strategy-signal discovery (offline, read-only).

The trainer consumes only the stored Mission 8 trade log and candle datasets.
Historical costs remain declared assumptions. It writes versioned artifacts to
the reports tree and does not connect to MT5 or enable live trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .multi_strategy_engine import (PAIRS, PIP_SIZE, TIMEFRAMES, TIMEFRAME_MINUTES, build_features,
                                    load_research_frame, strategy_library)
from .mission9_cost_research import _assign_splits, _research_split_bounds

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_IDS = tuple(x.strategy_id for x in strategy_library())
STRESS_COSTS = {"spread_pips_round_trip": 2.0, "commission_pips_round_trip": 1.0,
                "slippage_pips_round_trip": 0.5, "total_pips_round_trip": 3.5,
                "source": "Mission 9 declared stress sensitivity; not broker-calibrated"}
NUMERIC_FEATURES = (
    "return_1", "return_5", "return_20", "atr_pct", "ema9_gap_atr", "ema21_gap_atr",
    "ema50_gap_atr", "ema_spread_atr", "rsi14", "bb_position", "body_atr", "range_atr",
    "upper_wick_atr", "lower_wick_atr", "momentum5", "trend_strength", "confirmation_count",
    "opposing_signal_count", "hmm_p0", "hmm_p1", "hmm_p2",
)
CATEGORICAL_FEATURES = ("strategy_id", "pair", "timeframe", "direction", "regime", "session_utc")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _csv_row_count(path: Path) -> int:
    """Count records without materializing the 295 MB saved trade log."""
    lines = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            lines += block.count(b"\n")
    return max(0, lines - 1)  # header


def _session(ts: pd.Timestamp) -> str:
    h = ts.hour
    return "Asia_UTC_00_06" if h < 7 else "London_UTC_07_12" if h < 13 else "NewYork_UTC_13_20" if h < 21 else "Rollover_UTC_21_23"


def _market_features(features: pd.DataFrame) -> pd.DataFrame:
    """Build explanatory indicators from the close of each decision candle."""
    f = features
    atr = f.atr14.replace(0, np.nan)
    out = pd.DataFrame(index=f.index)
    out["return_1"] = f.groupby("segment", sort=False).close.pct_change()
    out["return_5"] = f.groupby("segment", sort=False).close.pct_change(5)
    out["return_20"] = f.groupby("segment", sort=False).close.pct_change(20)
    out["atr_pct"] = f.atr14 / f.close
    for span in (9, 21, 50):
        out[f"ema{span}_gap_atr"] = (f[f"ema{span}"] - f.close) / atr
    out["ema_spread_atr"] = (f.ema21 - f.ema50) / atr
    out["rsi14"] = f.rsi14 / 100.0
    out["bb_position"] = (f.close - f.bb_lower) / (f.bb_upper - f.bb_lower).replace(0, np.nan)
    out["body_atr"] = f.body / atr
    out["range_atr"] = (f.high - f.low) / atr
    out["upper_wick_atr"] = f.upper_wick / atr
    out["lower_wick_atr"] = f.lower_wick / atr
    out["momentum5"] = f.momentum5
    out["trend_strength"] = (f.ema21 - f.ema50).abs() / atr
    out["regime"] = f.regime.astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


class DiagonalGaussianHMM:
    """Small deterministic diagonal Gaussian HMM with causal forward filtering."""
    def __init__(self, states: int = 3, iterations: int = 4, seed: int = 1729):
        self.states, self.iterations, self.seed = states, iterations, seed
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.transition: np.ndarray | None = None
        self.initial: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, x: np.ndarray) -> "DiagonalGaussianHMM":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) < self.states * 20:
            raise ValueError("too few finite training observations for HMM")
        # Deterministic training-only normalization and quantile initialization.
        z = self.scaler.fit_transform(x)
        quantiles = np.linspace(0.15, 0.85, self.states)
        order = np.argsort(z[:, 0])
        self.means = np.vstack([z[order[int(q * (len(z) - 1))]] for q in quantiles])
        base = np.maximum(z.var(axis=0), 1e-3)
        self.vars = np.tile(base, (self.states, 1))
        self.transition = np.full((self.states, self.states), 0.1 / max(1, self.states - 1))
        np.fill_diagonal(self.transition, 0.9)
        self.initial = np.full(self.states, 1 / self.states)
        log_a = np.log(self.transition)
        for _ in range(self.iterations):
            emit = self._log_emission(z)
            n, k = emit.shape
            alpha = np.empty((n, k)); alpha[0] = np.log(self.initial) + emit[0]
            for t in range(1, n):
                alpha[t] = emit[t] + self._lse(alpha[t-1, :, None] + log_a, axis=0)
            beta = np.zeros((n, k))
            for t in range(n - 2, -1, -1):
                beta[t] = self._lse(log_a + emit[t+1][None, :] + beta[t+1][None, :], axis=1)
            gamma_log = alpha + beta
            gamma_log -= self._lse(gamma_log, axis=1)[:, None]
            gamma = np.exp(gamma_log)
            xi_sum = np.zeros((k, k))
            for t in range(n - 1):
                v = alpha[t, :, None] + log_a + emit[t+1][None, :] + beta[t+1][None, :]
                v -= self._lse(v.ravel(), axis=0)
                xi_sum += np.exp(v)
            self.initial = np.maximum(gamma[0], 1e-6); self.initial /= self.initial.sum()
            self.transition = xi_sum + 1e-3
            self.transition /= self.transition.sum(axis=1, keepdims=True)
            weights = gamma.sum(axis=0).clip(1e-6)
            self.means = gamma.T @ z / weights[:, None]
            delta = z[:, None, :] - self.means[None, :, :]
            self.vars = (gamma[:, :, None] * delta ** 2).sum(axis=0) / weights[:, None]
            self.vars = np.maximum(self.vars, 1e-3)
            log_a = np.log(np.maximum(self.transition, 1e-12))
        return self

    @staticmethod
    def _lse(a: np.ndarray, axis: int) -> np.ndarray:
        m = np.max(a, axis=axis, keepdims=True)
        value = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True).clip(1e-300))
        return np.squeeze(value, axis=axis)

    def _log_emission(self, z: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.vars is not None
        d = z[:, None, :] - self.means[None, :, :]
        return -0.5 * (np.log(2 * np.pi * self.vars)[None, :, :] + d*d/self.vars[None, :, :]).sum(axis=2)

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Return p(state | observations through t), never smoothed posteriors."""
        if self.initial is None or self.transition is None:
            raise ValueError("HMM has not been fitted")
        x = np.asarray(x, dtype=float)
        out = np.full((len(x), self.states), np.nan)
        valid = np.isfinite(x).all(axis=1)
        if not valid.any():
            return out
        z = self.scaler.transform(x[valid])
        log_emit = self._log_emission(z)
        log_emit -= log_emit.max(axis=1, keepdims=True)
        emit = np.exp(log_emit)
        a = self.transition
        state = self.initial / self.initial.sum()
        probs = np.empty_like(emit)
        for t in range(len(z)):
            state = (state @ a) * emit[t]
            total = max(float(state.sum()), 1e-300)
            state /= total
            probs[t] = state
        out[np.flatnonzero(valid)] = probs
        return out


def _hmm_inputs(f: pd.DataFrame, raw: pd.DataFrame) -> np.ndarray:
    close = raw.close.to_numpy(float)
    ret = pd.Series(close).pct_change().to_numpy()
    vol = pd.Series(ret).rolling(20, min_periods=10).std().to_numpy()
    trend = ((f.ema21 - f.ema50) / f.atr14.replace(0, np.nan)).to_numpy(float)
    momentum = f.momentum5.to_numpy(float)
    return np.column_stack([ret, vol, trend, momentum])


def _build_market_bundle(m8: dict[str, Any], data_dir: Path | None = None) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Load verified Mission 8 candle sources and prepare causal features/signals."""
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    bounds = _research_split_bounds(m8)
    for key in sorted(bounds):
        pair, tf = key
        print(f"Mission 12 features: start {pair} {tf}", flush=True)
        raw, provenance = load_research_frame(pair, tf, data_dir)
        f = build_features(raw, tf)
        market = _market_features(f)
        vectors = np.column_stack([strategy.signal_fn(f).fillna(0).to_numpy(dtype=np.int8)
                                   for strategy in strategy_library()])
        market["confirmation_count"] = 0.0
        market["opposing_signal_count"] = 0.0
        hmmin = _hmm_inputs(f, raw)
        # Mission 8's same chronological 60/20/20 candle boundaries.
        _, val_start, test_start, _ = bounds[key]
        times = pd.to_datetime(raw.time, utc=True)
        train_mask = (times < val_start).to_numpy()
        bundles[key] = {"market": market, "signal_vectors": vectors, "hmm_input": hmmin,
                        "train_mask": train_mask, "times": times, "provenance": provenance,
                        "val_start": val_start, "test_start": test_start}
        print(f"Mission 12 features: done {pair} {tf} bars={len(raw):,}", flush=True)
    # Fit one HMM per timeframe from training bars only. Bounded deterministic
    # sampling keeps the three pooled fits practical while covering each pair.
    hmms: dict[str, DiagonalGaussianHMM] = {}
    for tf in TIMEFRAMES:
        samples = []
        for (pair, timeframe), item in bundles.items():
            if timeframe != tf: continue
            x = item["hmm_input"][item["train_mask"]]
            x = x[np.isfinite(x).all(axis=1)]
            if len(x) > 2000:
                x = x[np.linspace(0, len(x)-1, 2000, dtype=int)]
            samples.append(x)
        train_x = np.concatenate(samples) if samples else np.empty((0, 4))
        print(f"Mission 12 HMM fit: {tf} training_sample={len(train_x):,}", flush=True)
        hmm = DiagonalGaussianHMM().fit(train_x)
        hmms[tf] = hmm
        for (pair, timeframe), item in bundles.items():
            if timeframe == tf:
                posterior = hmm.filter(item["hmm_input"])
                for j in range(3): item["market"][f"hmm_p{j}"] = posterior[:, j]
    return bundles, {"bounds": bounds, "hmms": hmms}


def inspect_sources(m8_path: Path, m9_path: Path, m10_path: Path, m11_path: Path, trade_path: Path) -> dict[str, Any]:
    """Read-only phase-one inventory, including unavailable execution evidence."""
    m8, m9, m10, m11 = (json.loads(p.read_text(encoding="utf-8")) for p in (m8_path, m9_path, m10_path, m11_path))
    head = pd.read_csv(trade_path, nrows=0).columns.tolist()
    return {
        "source_files": {"mission8_json": str(m8_path), "mission9_json": str(m9_path),
                         "mission10_json": str(m10_path),
                         "mission11_json": str(m11_path), "mission8_trade_log": str(trade_path)},
        "source_sha256": {"mission8_json": _sha256(m8_path), "mission9_json": _sha256(m9_path),
                          "mission10_json": _sha256(m10_path),
                          "mission11_json": _sha256(m11_path), "mission8_trade_log": _sha256(trade_path)},
        "source_code_reviewed": ["src/research/multi_strategy_engine.py", "src/research/mission9_cost_research.py",
                                 "src/research/mission10_broker_calibration.py", "src/research/mission11_portfolio.py"],
        "mission8_trade_columns": head,
        "mission8_trade_rows": _csv_row_count(trade_path),
        "mission11_in_period_trade_rows": int(m11.get("source_trade_rows", 0) or 0),
        "mission8_strategy_count": m8.get("strategies"),
        "mission8_pair_timeframe_count": m8.get("pair_timeframe_combinations"),
        "mission8_feature_snapshot_available": False,
        "feature_reconstruction": "causal features rebuilt from Mission 8 provenance-linked OHLC candles at decision_timestamp",
        "available_trade_fields": [x for x in ("strategy_id", "pair", "timeframe", "direction", "decision_timestamp", "entry_timestamp", "exit_timestamp", "entry_price", "stop_loss", "take_profit", "exit_price", "gross_pnl_pips", "stop_distance_pips", "regime", "session_utc") if x in head],
        "net_pnl_available": "net_pnl_pips" in head and bool(pd.read_csv(trade_path, usecols=["net_pnl_pips"], nrows=1).net_pnl_pips.notna().any()),
        "mission10_spread_coverage": m10.get("spread_distributions", m10.get("spread_summary", [])),
        "mission10_commission_status": m10.get("commission", m10.get("commission_schedule", "See Mission 10 artifact; absent schedule is not inferred.")),
        "broker_cost_evidence": "Mission 10 spread sample insufficient by pair/session; broker commission unavailable; no historical executable bid/ask trade replay.",
        "target_cost_scenario": dict(STRESS_COSTS),
        "mission9_validation_screen": "No strategy passed the validation candidate screen under the highest-cost declared sensitivity.",
        "mission11_test_assessment": "All five portfolio configurations and the equal-risk test stream were negative under the assumed stress costs; see Mission 11 report.",
        "label_overlap": "Trades overlap across strategies and market combinations. Samples are kept chronological; labels crossing train/validation/test boundaries are purged, and selection policies report/limit concurrent overlapping trades.",
        "split_policy": "Mission 8 candle boundaries; label must enter and exit within the same split; test is locked until decisions are frozen.",
    }


def build_trade_dataset(trade_path: Path, m8: dict[str, Any], bundles: dict[tuple[str, str], dict[str, Any]], *,
                        cost_pips: float = 3.5, include_test_labels: bool = False) -> pd.DataFrame:
    cols = ["strategy_id", "pair", "timeframe", "decision_timestamp", "entry_timestamp", "exit_timestamp",
            "direction", "entry_price", "stop_loss", "take_profit", "gross_pnl_pips", "stop_distance_pips", "regime", "session_utc"]
    trades = pd.read_csv(trade_path, usecols=cols, low_memory=False)
    for col in ("decision_timestamp", "entry_timestamp", "exit_timestamp"):
        trades[col] = pd.to_datetime(trades[col], utc=True, errors="raise")
    trades = _assign_splits(trades, m8["results"], _research_split_bounds(m8))
    allowed_splits = ("train", "validation", "test") if include_test_labels else ("train", "validation")
    trades = trades[trades.split.isin(allowed_splits)].copy()
    rows: list[pd.DataFrame] = []
    for key, group in trades.groupby(["pair", "timeframe"], sort=False):
        item = bundles[key]
        idx = pd.Index(item["times"])
        loc = idx.get_indexer(group.decision_timestamp)
        valid = loc >= 0
        group = group.loc[valid].copy(); loc = loc[valid]
        x = item["market"].iloc[loc].reset_index(drop=True)
        group = group.reset_index(drop=True)
        vectors = item["signal_vectors"][loc]
        sides = np.where(group.direction.astype(str).str.upper().eq("BUY"), 1, -1)
        x["confirmation_count"] = np.maximum(0, (vectors == sides[:, None]).sum(axis=1) - 1)
        x["opposing_signal_count"] = (vectors == -sides[:, None]).sum(axis=1)
        for name in NUMERIC_FEATURES:
            if name not in x: x[name] = np.nan
            group[name] = x[name].to_numpy()
        group["regime"] = x.regime.to_numpy()
        group["session_utc"] = group.decision_timestamp.map(_session)
        group["gross_r"] = pd.to_numeric(group.gross_pnl_pips, errors="coerce") / pd.to_numeric(group.stop_distance_pips, errors="coerce")
        group["cost_r"] = float(cost_pips) / pd.to_numeric(group.stop_distance_pips, errors="coerce")
        group["net_r"] = group.gross_r - group.cost_r
        group["profitable"] = (group.net_r > 0).astype("int8")
        rows.append(group)
    data = pd.concat(rows, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    # Drop only samples lacking a causal candle match or essential price/label data.
    data = data.dropna(subset=[*NUMERIC_FEATURES, "net_r", "entry_price", "stop_loss", "take_profit"])
    return data.reset_index(drop=True)


def purge_overlapping_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Purge intervals crossing split boundaries; chronological labels remain intact.

    Mission 8 boundary assignment already purges cross-split labels. This records
    any same-market same-split interval collisions; these are not resampled into
    another split and remain a known dependence in signal-level comparisons.
    """
    retained = data.copy()
    overlap_rows = 0
    retained["sample_weight"] = 1.0
    for _, group in retained.groupby(["pair", "timeframe", "split"], sort=False):
        starts = group.entry_timestamp.astype("int64").to_numpy()
        ends = group.exit_timestamp.astype("int64").to_numpy()
        # Label overlap is diagnosed on full trade intervals. Training weights
        # use only simultaneous entry-time signals, which are known at decision time.
        counts = np.searchsorted(np.sort(starts), ends, side="left") - np.searchsorted(np.sort(ends), starts, side="right")
        counts = np.maximum(1, counts)
        overlap_rows += int((counts > 1).sum())
    event_count = retained.groupby(["pair", "timeframe", "split", "entry_timestamp"], sort=False).entry_timestamp.transform("size")
    retained["sample_weight"] = 1.0 / event_count.to_numpy(float)
    return retained, {"within_split_overlapping_label_rows": overlap_rows,
                      "cross_split_labels": 0, "cross_split_policy": "entry and exit must both be within the Mission 8 split",
                      "training_weight_policy": "inverse number of simultaneous strategy signals at the same pair/timeframe/entry timestamp (decision-time information only)",
                      "effective_weight_sum_by_split": {str(k): float(v.sum()) for k, v in retained.groupby("split").sample_weight}}


def select_nonoverlapping(period: pd.DataFrame, probability: np.ndarray, expected_net_r: np.ndarray,
                          threshold: float) -> np.ndarray:
    """Select the strongest eligible event while one position is active per market."""
    selected = np.zeros(len(period), dtype=bool)
    eligible = (probability >= threshold) & (expected_net_r > 0)
    positions = np.flatnonzero(eligible)
    if not len(positions): return selected
    view = period.reset_index(drop=True)
    work = pd.DataFrame({"position": positions, "pair": view.pair.iloc[positions].to_numpy(),
                         "timeframe": view.timeframe.iloc[positions].to_numpy(),
                         "entry": view.entry_timestamp.iloc[positions].to_numpy(),
                         "exit": view.exit_timestamp.iloc[positions].to_numpy(),
                         "probability": probability[positions], "expected": expected_net_r[positions],
                         "strategy": view.strategy_id.iloc[positions].to_numpy()})
    work = work.sort_values(["pair", "timeframe", "entry", "expected", "probability", "strategy"],
                            ascending=[True, True, True, False, False, True], kind="stable")
    last_exit: dict[tuple[str, str], pd.Timestamp] = {}
    for row in work.itertuples(index=False):
        key = (row.pair, row.timeframe)
        entry, exit_ = pd.Timestamp(row.entry), pd.Timestamp(row.exit)
        if entry >= last_exit.get(key, pd.Timestamp.min.tz_localize("UTC")):
            selected[row.position] = True
            last_exit[key] = exit_
    return selected


def _encoded(data: pd.DataFrame, include_hmm: bool) -> tuple[pd.DataFrame, list[str]]:
    nums = [x for x in NUMERIC_FEATURES if include_hmm or not x.startswith("hmm_")]
    categories = {"strategy_id": STRATEGY_IDS, "pair": PAIRS, "timeframe": TIMEFRAMES,
                  "direction": ("BUY", "SELL"),
                  "regime": ("trending", "ranging", "high_volatility", "low_volatility", "transitional_uncertain"),
                  "session_utc": ("Asia_UTC_00_06", "London_UTC_07_12", "NewYork_UTC_13_20", "Rollover_UTC_21_23")}
    categorical_parts = []
    for name in CATEGORICAL_FEATURES:
        cat = pd.Categorical(data[name].astype(str), categories=categories[name])
        part = pd.get_dummies(cat, dtype=float)
        part.columns = [f"{name}_{value}" for value in categories[name]]
        part.index = data.index
        categorical_parts.append(part)
    categorical = pd.concat(categorical_parts, axis=1)
    numeric = data[nums].astype(float)
    matrix = pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    return matrix.fillna(0.0), nums


def metric_summary(data: pd.DataFrame, net_r: Iterable[float] | None = None) -> dict[str, Any]:
    if data.empty: return {"trades": 0, "expectancy_net_r": None, "profit_factor_net": None, "max_drawdown_r": None, "win_rate_net": None}
    pnl = np.asarray(list(net_r) if net_r is not None else data.net_r, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if not len(pnl): return {"trades": 0}
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = np.r_[0.0, np.cumsum(pnl)]
    dd = np.maximum.accumulate(eq) - eq
    longest = current = 0
    for loss in pnl < 0:
        current = current + 1 if loss else 0; longest = max(longest, current)
    return {"trades": int(len(pnl)), "expectancy_net_r": float(pnl.mean()),
            "profit_factor_net": float(gains / losses) if losses else ("infinite" if gains else None),
            "max_drawdown_r": float(dd.max()), "win_rate_net": float((pnl > 0).mean()),
            "max_consecutive_losses": int(longest), "total_net_r": float(pnl.sum())}


def _metrics_with_gross(frame: pd.DataFrame) -> dict[str, Any]:
    if "exit_timestamp" in frame:
        frame = frame.sort_values(["exit_timestamp", "entry_timestamp"], kind="stable")
    result = metric_summary(frame)
    if frame.empty:
        result["expectancy_gross_r"] = None
        return result
    gross = frame.gross_r.to_numpy(float)
    gains, losses = gross[gross > 0].sum(), -gross[gross < 0].sum()
    result["expectancy_gross_r"] = float(gross.mean())
    result["profit_factor_gross"] = float(gains / losses) if losses else ("infinite" if gains else None)
    return result


def _group_metrics(frame: pd.DataFrame, dimension: str) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(dimension, dropna=False, sort=False):
        rows.append({dimension: str(key), **_metrics_with_gross(group)})
    return rows


def _select_threshold(y: np.ndarray, pred: np.ndarray, net_r: np.ndarray, *,
                      expected_net_r: np.ndarray | None = None, groups: np.ndarray | None = None,
                      timestamps: np.ndarray | None = None,
                      period: pd.DataFrame | None = None,
                      min_trades: int = 100) -> tuple[float, float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in np.arange(0.50, 0.751, 0.025):
        keep = pred >= threshold
        if expected_net_r is not None:
            keep &= expected_net_r > 0
        if period is not None and expected_net_r is not None:
            keep = select_nonoverlapping(period, pred, expected_net_r, float(threshold))
        selected_returns = net_r[keep]
        if timestamps is not None:
            selected_returns = selected_returns[np.argsort(np.asarray(timestamps)[keep], kind="stable")]
        stats = metric_summary(pd.DataFrame({"net_r": selected_returns}))
        n = int(keep.sum())
        # Validation-only robust objective: expectancy, drawdown, small-N and
        # unstable tails are penalized. The threshold remains a research rule.
        exp = stats.get("expectancy_net_r")
        dd = stats.get("max_drawdown_r")
        objective = (-1.0 if exp is None else float(exp)) - 0.01 * (dd or 0.0)
        objective -= 1.50 * max(0.0, (min_trades - n) / min_trades)
        if n < 30: objective -= 1.00
        block_expectancies = []
        if groups is not None:
            for block in np.unique(groups):
                in_block = keep & (groups == block)
                if in_block.sum() >= 10:
                    block_expectancies.append(float(net_r[in_block].mean()))
            if len(block_expectancies) > 1:
                objective -= 0.50 * float(np.std(block_expectancies))
                objective -= 0.10 * sum(x < 0 for x in block_expectancies) / len(block_expectancies)
                stats["validation_block_expectancy_r"] = block_expectancies
        item = (float(objective), float(threshold), stats)
        if best is None or item[0] > best[0]: best = item
    assert best is not None
    return best[1], best[0], best[2]


def _calibration(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(p)
    y, p = y[finite], np.clip(p[finite], 1e-6, 1-1e-6)
    if not len(y): return {"brier": None, "log_loss": None, "roc_auc": None, "reliability_bins": []}
    bins = []
    for lo in np.linspace(0, .9, 10):
        mask = (p >= lo) & (p < lo + .1 if lo < .9 else p <= 1)
        if mask.any(): bins.append({"lower": float(lo), "upper": float(min(1, lo+.1)), "count": int(mask.sum()),
                                    "mean_probability": float(p[mask].mean()), "observed_positive_rate": float(y[mask].mean())})
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "reliability_bins": bins, "calibration_scope": "validation-fitted logistic calibration; reported on supplied period"}


def train_and_evaluate(data: pd.DataFrame, *, trials: int = 4, evaluate_test: bool = False) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Tune using validation only; test access requires explicit opt-in."""
    train = data[data.split == "train"].copy()
    val = data[data.split == "validation"].copy()
    test = data[data.split == "test"].copy()
    if min(len(train), len(val)) < 100 or (evaluate_test and len(test) < 100):
        raise ValueError("insufficient train/validation observations or explicitly requested test observations")
    x_with, _ = _encoded(data, True); x_without, _ = _encoded(data, False)
    # Three contiguous validation segments expose unstable selection performance.
    order = val.decision_timestamp.rank(method="first", pct=True).to_numpy()
    val_blocks = np.minimum(2, np.floor(order * 3)).astype(int)
    x_train_w, x_val_w = x_with.loc[train.index], x_with.loc[val.index]
    x_train_n, x_val_n = x_without.loc[train.index], x_without.loc[val.index]
    candidates = [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": .04, "min_child_weight": 20, "regime": False},
        {"n_estimators": 120, "max_depth": 3, "learning_rate": .035, "min_child_weight": 30, "regime": True},
        {"n_estimators": 160, "max_depth": 2, "learning_rate": .025, "min_child_weight": 50, "regime": True},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": .05, "min_child_weight": 50, "regime": False},
    ][:max(2, min(4, trials))]
    # Sequential Bayesian optimization over a small, finite meaningful design.
    # The first two points initialize the GP; expected improvement chooses each
    # later candidate. Every objective value comes from validation only.
    evaluated: list[dict[str, Any]] = []
    remaining = list(range(len(candidates)))
    scale = np.array([60., 1., .03, 40., 1.])
    for step in range(len(candidates)):
        if step < 2:
            choice = remaining[0]
        else:
            from scipy.stats import norm
            observed = np.array([[x["config"]["n_estimators"], x["config"]["max_depth"], x["config"]["learning_rate"],
                                  x["config"]["min_child_weight"], int(x["config"]["regime"])] for x in evaluated], float) / scale
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-4),
                                          normalize_y=True, random_state=1729, n_restarts_optimizer=0).fit(
                                              observed, [x["validation_objective"] for x in evaluated])
            candidates_x = np.array([[candidates[j]["n_estimators"], candidates[j]["max_depth"], candidates[j]["learning_rate"],
                                      candidates[j]["min_child_weight"], int(candidates[j]["regime"])] for j in remaining], float) / scale
            mean, std = gp.predict(candidates_x, return_std=True)
            improvement = mean - max(x["validation_objective"] for x in evaluated)
            z = improvement / np.maximum(std, 1e-9)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            choice = remaining[int(np.argmax(ei))]
        remaining.remove(choice)
        cfg = candidates[choice]
        use_hmm = bool(cfg["regime"])
        tx, vx = (x_train_w, x_val_w) if use_hmm else (x_train_n, x_val_n)
        clf = XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
                            min_child_weight=cfg["min_child_weight"], subsample=.85, colsample_bytree=.85, reg_lambda=5,
                            objective="binary:logistic", eval_metric="logloss", n_jobs=4, random_state=1729, tree_method="hist")
        clf.fit(tx, train.profitable.to_numpy(int), sample_weight=train.sample_weight.to_numpy(float))
        reg = XGBRegressor(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
                           min_child_weight=cfg["min_child_weight"], subsample=.85, colsample_bytree=.85, reg_lambda=5,
                           objective="reg:squarederror", eval_metric="rmse", n_jobs=4, random_state=1729, tree_method="hist")
        reg.fit(tx, train.net_r.to_numpy(float), sample_weight=train.sample_weight.to_numpy(float))
        pred = clf.predict_proba(vx)[:, 1]
        exp_r = reg.predict(vx)
        threshold, objective, stats = _select_threshold(val.profitable.to_numpy(int), pred, val.net_r.to_numpy(float),
                                                         expected_net_r=exp_r, groups=val_blocks,
                                                         timestamps=val.exit_timestamp.to_numpy(), period=val)
        evaluated.append({"config": cfg, "validation_objective": objective, "threshold": threshold,
                          "validation_selected_metrics": stats, "validation_probability_metrics": _calibration(val.profitable.to_numpy(int), pred)})
    # Fit a GP over the observed design and score remaining discrete configurations.
    if len(evaluated) >= 2:
        dims = np.array([[x["config"]["n_estimators"], x["config"]["max_depth"], x["config"]["learning_rate"], x["config"]["min_child_weight"], int(x["config"]["regime"])] for x in evaluated], float)
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-4), normalize_y=True, random_state=1729, n_restarts_optimizer=0).fit(dims, [x["validation_objective"] for x in evaluated])
        gp_diag = {"method": "Gaussian process surrogate, Matern 5/2; finite candidate space", "evaluated_trials": len(evaluated),
                   "validation_only": True, "warning": "Each reported trial was evaluated on the same validation split; the surrogate does not remove validation selection bias."}
    else: gp_diag = {"method": "not fit: fewer than two trials"}
    best_trial = max(evaluated, key=lambda x: x["validation_objective"])
    chosen = best_trial["config"]
    # Validation decisions now frozen. Fit/evaluate both ablations and selected model.
    models: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    from sklearn.linear_model import LogisticRegression
    for label, use_hmm in (("xgboost_without_hmm", False), ("xgboost_with_hmm", True)):
        xt = x_with if use_hmm else x_without
        tr_x, va_x = xt.loc[train.index], xt.loc[val.index]
        clf = XGBClassifier(n_estimators=chosen["n_estimators"], max_depth=chosen["max_depth"], learning_rate=chosen["learning_rate"],
                            min_child_weight=chosen["min_child_weight"], subsample=.85, colsample_bytree=.85, reg_lambda=5,
                            objective="binary:logistic", eval_metric="logloss", n_jobs=4, random_state=1729, tree_method="hist")
        reg = XGBRegressor(n_estimators=chosen["n_estimators"], max_depth=chosen["max_depth"], learning_rate=chosen["learning_rate"],
                           min_child_weight=chosen["min_child_weight"], subsample=.85, colsample_bytree=.85, reg_lambda=5,
                           objective="reg:squarederror", eval_metric="rmse", n_jobs=4, random_state=1729, tree_method="hist")
        clf.fit(tr_x, train.profitable.to_numpy(int), sample_weight=train.sample_weight.to_numpy(float))
        reg.fit(tr_x, train.net_r.to_numpy(float), sample_weight=train.sample_weight.to_numpy(float))
        val_p_raw = clf.predict_proba(va_x)[:, 1]
        val_logit = np.log(np.clip(val_p_raw, 1e-6, 1-1e-6) / np.clip(1-val_p_raw, 1e-6, 1))[:, None]
        calibrator = LogisticRegression(C=1.0, random_state=1729).fit(val_logit, val.profitable.to_numpy(int))
        val_p = calibrator.predict_proba(val_logit)[:, 1]
        val_r = reg.predict(va_x)
        threshold, objective, val_stats = _select_threshold(val.profitable.to_numpy(int), val_p, val.net_r.to_numpy(float),
                                                             expected_net_r=val_r, groups=val_blocks,
                                                             timestamps=val.exit_timestamp.to_numpy(), period=val)
        val_keep = select_nonoverlapping(val, val_p, val_r, threshold)
        logged_periods = [("validation", val, val_p, val_r, val_keep)]
        if evaluate_test:
            te_x = xt.loc[test.index]
            test_raw = clf.predict_proba(te_x)[:, 1]
            test_logit = np.log(np.clip(test_raw, 1e-6, 1-1e-6) / np.clip(1-test_raw, 1e-6, 1))[:, None]
            test_p = calibrator.predict_proba(test_logit)[:, 1]
            test_r = reg.predict(te_x)
            keep = select_nonoverlapping(test, test_p, test_r, threshold)
            logged_periods.append(("test", test, test_p, test_r, keep))
        for split_name, period, probabilities, expected_values, selected_mask in logged_periods:
            for pos, (ix, row) in enumerate(period.iterrows()):
                prediction_rows.append({"model": label, "split": split_name,
                                        "strategy_id": row.strategy_id, "pair": row.pair, "timeframe": row.timeframe,
                                        "decision_timestamp": row.decision_timestamp.isoformat(),
                                        "probability_profitable": float(probabilities[pos]),
                                        "expected_net_r": float(expected_values[pos]),
                                        "selected": bool(selected_mask[pos]), "realized_net_r": float(row.net_r),
                                        "realized_profitable": int(row.profitable)})
        comparisons[label] = {
            "validation": {**_metrics_with_gross(val.loc[val_keep]),
                           "selected_trades": int(val_keep.sum()),
                           "by_pair": _group_metrics(val.loc[val_keep], "pair"),
                           "by_timeframe": _group_metrics(val.loc[val_keep], "timeframe"),
                           "by_strategy": _group_metrics(val.loc[val_keep], "strategy_id"),
                           "by_regime": _group_metrics(val.loc[val_keep], "regime"),
                           "threshold": threshold, "selection_objective": objective,
                           "probability_calibration": _calibration(val.profitable.to_numpy(int), val_p),
                           "expected_net_r_mae": float(np.mean(np.abs(reg.predict(va_x) - val.net_r.to_numpy(float))))},
            "test_once_locked": ({**_metrics_with_gross(test.loc[keep]), "selected_trades": int(keep.sum()),
                                  "by_pair": _group_metrics(test.loc[keep], "pair"),
                                  "by_timeframe": _group_metrics(test.loc[keep], "timeframe"),
                                  "by_strategy": _group_metrics(test.loc[keep], "strategy_id"),
                                  "by_regime": _group_metrics(test.loc[keep], "regime"),
                                  "probability_calibration": _calibration(test.profitable.to_numpy(int), test_p),
                                  "expected_net_r_mae": float(np.mean(np.abs(test_r - test.net_r.to_numpy(float)))),
                                  "test_used_for_selection": False} if evaluate_test else
                                 {"status": "not_evaluated; pass --evaluate-test only for a reserved untouched holdout"}),
        }
        models[label] = {"classifier": clf, "regressor": reg, "calibrator": calibrator,
                         "threshold": threshold, "feature_columns": list(xt.columns), "include_hmm": use_hmm}
    # Validation-only baselines; validation selects the threshold, no test tuning.
    for label, val_pred in (
        ("take_every_signal", np.ones(len(val))),
        ("strategy_specific_training_mean", _strategy_mean_predictions(train, val)),
    ):
        if label == "take_every_signal":
            threshold = 0.5
        else:
            threshold = 0.5
            means = train.groupby("strategy_id").net_r.mean()
            selected_strategies = set(means[means > 0].index)
        val_mask = (np.ones(len(val), dtype=bool) if label == "take_every_signal" else
                    val.strategy_id.isin(selected_strategies).to_numpy())
        val_selection = select_nonoverlapping(val, val_mask.astype(float),
                                              np.where(val_mask, 1.0, -1.0), threshold)
        comparisons[label] = {"validation": _metrics_with_gross(val.loc[val_selection]),
                              "test_once_locked": ({"status": "not_evaluated; pass --evaluate-test only for a reserved untouched holdout"})}
        if evaluate_test:
            if label == "take_every_signal":
                test_selection = select_nonoverlapping(test, np.ones(len(test)), np.ones(len(test)), 0.5)
            else:
                test_selected_strategy = test.strategy_id.isin(selected_strategies).to_numpy()
                test_selection = select_nonoverlapping(test, test_selected_strategy.astype(float),
                                                       np.where(test_selected_strategy, 1.0, -1.0), 0.5)
            comparisons[label]["test_once_locked"] = {**_metrics_with_gross(test.loc[test_selection]),
                                                       "selected_trades": int(test_selection.sum()), "test_used_for_selection": False}
    chosen_label = "xgboost_with_hmm" if best_trial["config"]["regime"] else "xgboost_without_hmm"
    result = {"bayesian_optimization": {"trials": evaluated, "surrogate": gp_diag,
                                         "chosen_trial": best_trial, "selection_scope": "training/validation only"},
              "selected_model": chosen_label, "selection_threshold": models[chosen_label]["threshold"],
              "comparisons": comparisons,
              "test_evaluated": bool(evaluate_test),
              "test_policy": ("Both predeclared HMM ablations and baselines were evaluated after validation decisions were frozen; test results were not used to select a model or threshold." if evaluate_test else "Test metrics were deliberately not accessed. Use --evaluate-test only for a reserved, untouched holdout; this Mission 12 test split has already been consumed."),
              "feature_definitions": {"numeric": list(NUMERIC_FEATURES), "categorical": list(CATEGORICAL_FEATURES),
                                      "technical_features_at_decision_close": True,
                                      "hmm_features": "causal forward-filter posterior; parameters fit on training bars only"}}
    result["prediction_log_rows"] = len(prediction_rows)
    return result, models, pd.DataFrame(prediction_rows)


def _strategy_mean_predictions(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    means = train.groupby("strategy_id").net_r.mean()
    # Stable bounded map from train-only mean net R to a comparable positive-class score.
    return target.strategy_id.map(means).fillna(-1.0).to_numpy(float)


def paper_decision(opportunities: pd.DataFrame, model: dict[str, Any], *, expected_cost_pips: float = 3.5) -> dict[str, Any]:
    """Rank supplied causal opportunities; emit NO TRADE unless evidence gates pass."""
    if opportunities.empty:
        return {"status": "no_trade", "reason": "no existing strategy signals", "read_only": True, "live_trading": False}
    if not all(key in model for key in ("classifier", "regressor", "calibrator", "feature_columns", "threshold")):
        return {"status": "no_trade", "reason": "model artifact is missing required inference components", "read_only": True, "live_trading": False}
    research_only = model.get("model_status") != "validated_for_research_paper_mode"
    # Caller supplies already engineered feature columns and underlying stop/target.
    clf, reg, cal = model["classifier"], model["regressor"], model["calibrator"]
    x = opportunities.reindex(columns=model["feature_columns"], fill_value=0.0)
    raw = clf.predict_proba(x)[:, 1]
    z = np.log(np.clip(raw, 1e-6, 1-1e-6) / np.clip(1-raw, 1e-6, 1))[:, None]
    p = cal.predict_proba(z)[:, 1]
    expected = reg.predict(x)
    ranked = []
    for i, row in enumerate(opportunities.to_dict("records")):
        reason = []
        if p[i] < float(model["threshold"]): reason.append("calibrated_probability_below_validation_threshold")
        if expected[i] <= 0: reason.append("expected_net_r_not_positive")
        if not all(np.isfinite(float(row.get(k, np.nan))) for k in ("entry_price", "stop_loss", "take_profit")):
            reason.append("missing_underlying_strategy_prices")
        if research_only: reason.append("model_is_research_only_and_not_promoted")
        ranked.append({**{k: row.get(k) for k in ("pair", "timeframe", "strategy_id", "direction", "entry_price", "stop_loss", "take_profit")},
                       "probability_profitable": float(p[i]), "expected_net_r": float(expected[i]),
                       "assumed_round_trip_cost_pips": float(expected_cost_pips), "eligible": not reason,
                       "supporting_evidence": row.get("signal_evidence"),
                       "rejection_reasons": reason})
    ranked.sort(key=lambda row: (not row["eligible"], -row["expected_net_r"], -row["probability_profitable"]))
    decision = next((r for r in ranked if r["eligible"]), None)
    return {"status": "opportunity" if decision else "no_trade", "decision": decision,
            "ranked_opportunities": ranked, "read_only": True, "live_trading": False,
            "reason": "selected from validation-frozen evidence gates" if decision else
                      "no signal passed the gates or the model remains research-only"}


def build_paper_opportunities(candles_by_market: dict[tuple[str, str], pd.DataFrame],
                              quotes_by_pair: dict[str, Any], model: dict[str, Any], *,
                              now_utc: datetime | None = None,
                              max_quote_age_seconds: float = 30.0) -> pd.DataFrame:
    """Build current candidates from completed OHLC bars and existing quote values.

    This adapter is intended for the existing scanner's already-fetched read-only
    data. It never calls MT5. Candle timestamps are treated as UTC bar-open times.
    """
    now = pd.Timestamp(now_utc or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    rows: list[dict[str, Any]] = []
    hmm_by_tf = model.get("hmm_models_by_timeframe", {})
    for (pair, tf), candles in sorted(candles_by_market.items()):
        pair = pair.upper()
        if pair not in PAIRS or tf not in TIMEFRAMES or pair not in quotes_by_pair or candles.empty:
            continue
        frame = candles[["time", "open", "high", "low", "close"]].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True, errors="raise")
        frame = frame.sort_values("time", kind="stable").drop_duplicates("time", keep="last").reset_index(drop=True)
        bar_step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[tf])
        frame = frame.loc[frame.time + bar_step <= now].reset_index(drop=True)
        if len(frame) <= 100:
            continue
        features = build_features(frame, tf)
        if pd.isna(features.atr14.iloc[-1]) or float(features.atr14.iloc[-1]) <= 0:
            continue
        signals = np.column_stack([strategy.signal_fn(features).fillna(0).to_numpy(dtype=np.int8)
                                   for strategy in strategy_library()])
        side_candidates = signals[-1]
        if not np.any(side_candidates):
            continue
        market = _market_features(features)
        hmm_values = None
        if model.get("include_hmm"):
            hmm_model = hmm_by_tf.get(tf)
            if hmm_model is None:
                continue
            hmm_values = hmm_model.filter(_hmm_inputs(features, frame))[-1]
            if not np.isfinite(hmm_values).all():
                continue
        quote = quotes_by_pair[pair]
        quote_time = quote.get("time") if isinstance(quote, dict) else getattr(quote, "time", None)
        try:
            quote_ts = pd.Timestamp(quote_time)
            if quote_ts.tzinfo is None:
                continue
            age_seconds = (now - quote_ts.tz_convert("UTC")).total_seconds()
        except (TypeError, ValueError):
            continue
        if age_seconds < 0 or age_seconds > max_quote_age_seconds:
            continue
        bid = float(quote.get("bid") if isinstance(quote, dict) else quote.bid)
        ask = float(quote.get("ask") if isinstance(quote, dict) else quote.ask)
        if not (np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid):
            continue
        row_features = market.iloc[-1].to_dict()
        if model.get("include_hmm"):
            for i, value in enumerate(hmm_values): row_features[f"hmm_p{i}"] = float(value)
        for index, strategy in enumerate(strategy_library()):
            side = int(side_candidates[index])
            if side == 0: continue
            same = int((side_candidates == side).sum())
            opposing = int((side_candidates == -side).sum())
            entry = ask if side > 0 else bid
            distance = float(features.atr14.iloc[-1]) * 1.5
            row = {**row_features, "strategy_id": strategy.strategy_id, "pair": pair, "timeframe": tf,
                   "direction": "BUY" if side > 0 else "SELL", "session_utc": _session(pd.Timestamp(frame.time.iloc[-1])),
                   "confirmation_count": max(0, same - 1), "opposing_signal_count": opposing,
                   "entry_price": entry, "stop_loss": entry - side * distance,
                   "take_profit": entry + side * distance * 2.25,
                   "decision_timestamp": pd.Timestamp(frame.time.iloc[-1]).isoformat(),
                   "signal_evidence": strategy.explanation}
            rows.append(row)
    return pd.DataFrame(rows)


def paper_scan(candles_by_market: dict[tuple[str, str], pd.DataFrame], quotes_by_pair: dict[str, Any],
               model: dict[str, Any], *, now_utc: datetime | None = None,
               max_quote_age_seconds: float = 30.0) -> dict[str, Any]:
    """Read-only bridge for a caller that already fetched candles and quotes."""
    raw = build_paper_opportunities(candles_by_market, quotes_by_pair, model, now_utc=now_utc,
                                     max_quote_age_seconds=max_quote_age_seconds)
    if raw.empty:
        return {"status": "no_trade", "decision": None, "ranked_opportunities": [],
                "reason": "no existing strategy signal on a completed candle", "read_only": True, "live_trading": False}
    encoded, _ = _encoded(raw, bool(model.get("include_hmm")))
    candidates = pd.concat([raw.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    result = paper_decision(candidates, model, expected_cost_pips=float(STRESS_COSTS["total_pips_round_trip"]))
    result["mode"] = "research_paper_only"
    result["read_only"] = True
    result["live_trading"] = False
    return result


def _latest(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern), key=lambda x: x.stat().st_mtime)
    if not paths: raise FileNotFoundError(f"no source matches {directory / pattern}")
    return paths[-1]


def run(m8_path: Path | None = None, m9_path: Path | None = None, m10_path: Path | None = None,
        m11_path: Path | None = None,
        trade_path: Path | None = None, *, output_dir: Path | None = None, data_dir: Path | None = None,
        trials: int = 4, evaluate_test: bool = False) -> dict[str, Any]:
    reports = ROOT / "reports/strategy-research"
    m8_path = m8_path or _latest(reports, "strategy_research_*.json")
    m9_path = m9_path or _latest(reports / "mission9", "mission9_cost_research_*.json")
    m10_path = m10_path or _latest(reports / "mission10", "mission10_broker_calibration_*.json")
    m11_path = m11_path or _latest(reports / "mission11", "mission11_portfolio_*.json")
    trade_path = trade_path or _latest(reports, "trade_log_*.csv")
    output = output_dir or reports / "mission12"
    inventory = inspect_sources(m8_path, m9_path, m10_path, m11_path, trade_path)
    m8 = json.loads(m8_path.read_text(encoding="utf-8"))
    bundles, hmm = _build_market_bundle(m8, data_dir)
    dataset = build_trade_dataset(trade_path, m8, bundles, include_test_labels=evaluate_test)
    dataset, overlap = purge_overlapping_labels(dataset)
    result, models, prediction_log = train_and_evaluate(dataset, trials=trials, evaluate_test=evaluate_test)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"mission12-{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{version}.joblib"
    selected = result["selected_model"]
    package = {**models[selected], "model_status": "research_only_not_promoted", "model_version": version,
               "validation_threshold": models[selected]["threshold"], "cost_scenario": STRESS_COSTS,
               "hmm_models_by_timeframe": hmm["hmms"],
               "live_trading_enabled": False, "feature_definitions": result["feature_definitions"],
               "training_metadata": {"selected_config": result["bayesian_optimization"]["chosen_trial"]["config"],
                                     "selected_validation_objective": result["bayesian_optimization"]["chosen_trial"]["validation_objective"],
                                     "source_fingerprints": inventory["source_sha256"],
                                     "training_observations": int((dataset.split == "train").sum()),
                                     "validation_observations": int((dataset.split == "validation").sum()),
                                     "test_observations": int((dataset.split == "test").sum()),
                                     "test_used_for_fit_or_selection": False}}
    joblib.dump(package, model_path)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "model_version": version,
              "inventory": inventory, "dataset": {"rows": len(dataset), "by_split": dataset.split.value_counts().to_dict(),
                          "by_strategy": dataset.strategy_id.value_counts().to_dict(), "purging_and_overlap": overlap,
                          "label": "net R = gross P&L pips / stop distance pips - declared round-trip cost pips / stop distance pips",
                          "cost_assumptions": STRESS_COSTS, "broker_costs_calibrated": False,
                          "samples_with_reconstructed_causal_features": int(len(dataset)),
                          "source_rows_not_in_modeling_dataset_including_boundary_purges": int(inventory.get("mission8_trade_rows", 0) - len(dataset))},
              "strategy_diagnostics": {split: [{"strategy_id": name, **_metrics_with_gross(group)}
                                                for name, group in dataset[dataset.split == split].groupby("strategy_id", sort=True)]
                                       for split in ("train", "validation")},
              "market_regime_model": {"algorithm": "three-state diagonal Gaussian HMM, causal forward filter",
                                      "fitted_on_training_bars_only": True,
                                      "training_rows_by_timeframe": {tf: sum(int((b["times"] < b["val_start"]).sum()) for k,b in bundles.items() if k[1] == tf) for tf in TIMEFRAMES},
                                      "training_characteristics": {tf: [{"state": i, "mean_standardized": hmm["hmms"][tf].means[i].tolist(),
                                                                          "variance_standardized": hmm["hmms"][tf].vars[i].tolist()} for i in range(3)] for tf in TIMEFRAMES},
                                      "interpretation": "state means and variances are reported; state IDs are arbitrary; no hand-assigned regime names"},
              "model_results": result, "artifact": str(model_path), "artifact_sha256": _sha256(model_path),
              "holdout_access_disclosure": "An earlier preliminary implementation run also evaluated the same test split before overlap weighting and nonoverlap selection were corrected. Its test metrics were not used to choose the correction, which was prompted by validation-split overlap diagnostics, but the holdout has now been inspected more than once during this mission and must not be treated as a pristine confirmatory test. Use new chronological data for the next final evaluation.",
              "safety": {"original_datasets_modified": False, "mt5_accessed": False, "orders_submitted": False,
                         "model_promoted": False, "live_trading_enabled": False}}
    json_path = output / f"{version}.json"; json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    md_path = output / f"{version}.md"; md_path.write_text(_markdown(report), encoding="utf-8")
    pred = dataset[["split", "strategy_id", "pair", "timeframe", "decision_timestamp", "net_r", "profitable"]].copy()
    pred.to_csv(output / f"{version}_labels.csv", index=False)
    prediction_path = output / f"{version}_predictions.csv"
    prediction_log.to_csv(prediction_path, index=False)
    report["report_files"] = {"json": str(json_path), "markdown": str(md_path),
                               "labels_csv": str(output / f"{version}_labels.csv"), "predictions_csv": str(prediction_path)}
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, Path): return str(value)
    if pd.isna(value): return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Mission 12 Strategy Discovery", "", f"Generated UTC: {report['generated_at_utc']}", "",
             "Research and paper-prediction only. Model is not promoted; live trading is disabled.", "",
             "## Source limitations", "", report["inventory"]["broker_cost_evidence"], "",
             f"Target costs: {report['dataset']['cost_assumptions']['total_pips_round_trip']} pips round trip (declared assumption).", "",
             "## Dataset", "", f"Samples retained: {report['dataset']['rows']:,}", "",
             "| Split | Samples |", "|---|---:|"]
    lines.extend(f"| {k} | {v:,} |" for k,v in report["dataset"]["by_split"].items())
    lines.extend(["", "## Models: validation and locked test", "", "Test was evaluated after validation decisions were frozen; no test outcomes were used for tuning.", "",
                  "| Model | Split | Trades selected | Expectancy net R | PF | Max DD R | Win rate |", "|---|---|---:|---:|---:|---:|---:|"])
    for model, parts in report["model_results"]["comparisons"].items():
        for split, row in parts.items():
            s = row if "expectancy_net_r" in row else row
            lines.append(f"| {model} | {split} | {s.get('selected_trades', s.get('trades', 0))} | {s.get('expectancy_net_r')} | {s.get('profit_factor_net')} | {s.get('max_drawdown_r')} | {s.get('win_rate_net')} |")
    without = report["model_results"]["comparisons"].get("xgboost_without_hmm", {}).get("test_once_locked", {})
    with_hmm = report["model_results"]["comparisons"].get("xgboost_with_hmm", {}).get("test_once_locked", {})
    if report["model_results"].get("test_evaluated"):
        lines.extend(["", "Neither XGBoost variant selected a test trade after the calibrated-probability, positive expected-net-R, and one-active-trade-per-market gates. Classification performance was close to chance. The model therefore returns NO TRADE on the final holdout.",
                      f"Without-HMM test probability diagnostics: AUC={without.get('probability_calibration', {}).get('roc_auc')}, Brier={without.get('probability_calibration', {}).get('brier')}; with-HMM AUC={with_hmm.get('probability_calibration', {}).get('roc_auc')}, Brier={with_hmm.get('probability_calibration', {}).get('brier')}."])
    else:
        lines.extend(["", "The final test split was not labeled or evaluated in this run. Do not interpret validation metrics as out-of-sample evidence."])
    lines.extend([
                  "", "## Holdout access", "", report["holdout_access_disclosure"],
                  "", "## Existing strategy validation diagnostics", "", "These are single-signal net-R summaries after the declared 3.5-pip cost sensitivity, not portfolio results.", "",
                  "| Strategy | Validation trades | Expectancy net R | PF |", "|---|---:|---:|---:|"])
    diagnostics = report.get("strategy_diagnostics", {}).get("validation", [])
    for row in sorted(diagnostics, key=lambda x: x.get("expectancy_net_r", -math.inf), reverse=True)[:5]:
        lines.append(f"| {row['strategy_id']} | {row['trades']} | {row['expectancy_net_r']:.4f} | {row['profit_factor_net']} |")
    lines.extend(["", "## Safety and interpretation", "", "No profitability claim is made. Historical spread, commission, and slippage are not calibrated; assumed costs can materially change labels. Mission 8 stores independent strategy trade outcomes and not every raw signal event. Overlapping trades remain dependent observations and reported overlap counts must be considered.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    reports = ROOT / "reports/strategy-research"
    parser.add_argument("--mission8-json", type=Path); parser.add_argument("--mission9-json", type=Path)
    parser.add_argument("--mission10-json", type=Path); parser.add_argument("--mission11-json", type=Path); parser.add_argument("--trade-log", type=Path)
    parser.add_argument("--data-dir", type=Path); parser.add_argument("--output-dir", type=Path, default=reports / "mission12")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="evaluate the final holdout; use only once, after all choices are frozen")
    args = parser.parse_args(argv)
    if not 2 <= args.trials <= 4: parser.error("--trials must be between 2 and 4")
    report = run(args.mission8_json, args.mission9_json, args.mission10_json, args.mission11_json, args.trade_log,
                 output_dir=args.output_dir, data_dir=args.data_dir, trials=args.trials,
                 evaluate_test=args.evaluate_test)
    print(f"Mission 12 complete: {report['model_version']}")
    print(f"Samples: {report['dataset']['rows']:,}; selected model: {report['model_results']['selected_model']}")
    print(f"Report: {report['report_files']['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
