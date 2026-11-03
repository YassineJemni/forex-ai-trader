"""Purged training folds, serial-dependence-aware uncertainty, and append-only ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from .experiment_store import DEFAULT_DB_PATH, canonical_json


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    history_start_utc: str
    history_end_utc: str
    purge_start_utc: str
    evaluation_start_utc: str
    evaluation_end_utc: str
    embargo_end_utc: str
    history_rows: int
    purged_rows: int
    evaluation_rows: int
    embargo_rows: int
    purge_duration: str
    embargo_duration: str


def build_walk_forward_folds(
    timestamps: Sequence[Any] | pd.Series,
    *,
    outer_start: Any,
    outer_end: Any,
    maximum_outcome_horizon: str | pd.Timedelta,
    n_folds: int = 3,
    initial_history_fraction: float = 0.5,
    embargo: str | pd.Timedelta | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Build expanding-history folds inside one outer development period.

    Purge is the outcome horizon plus one bar for the strict next-open entry
    convention when the caller includes that bar in `maximum_outcome_horizon`.
    If omitted, embargo defaults to the same duration. Minute-aligned missing
    data are retained; counts describe actual candle rows in every interval.
    """
    values = pd.Series(timestamps)
    if values.empty or not isinstance(values.dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamps must be a nonempty timezone-aware series")
    values = values.dt.tz_convert("UTC")
    if values.isna().any() or values.duplicated().any() or not values.is_monotonic_increasing:
        raise ValueError("timestamps must be valid, unique, and ordered")
    start, end = pd.Timestamp(outer_start), pd.Timestamp(outer_end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("outer boundaries must be timezone-aware")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    purge = pd.Timedelta(maximum_outcome_horizon)
    embargo_delta = purge if embargo is None else pd.Timedelta(embargo)
    if purge <= pd.Timedelta(0) or embargo_delta < pd.Timedelta(0):
        raise ValueError("purge must be positive and embargo cannot be negative")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    if not 0 < initial_history_fraction < 1:
        raise ValueError("initial_history_fraction must be between zero and one")
    timeline = values.loc[(values >= start) & (values < end)]
    if len(timeline) < n_folds * 3:
        raise ValueError("outer period has too few candles for the requested folds")
    span = end - start
    initial_end = start + span * initial_history_fraction
    remaining = end - initial_end - purge * n_folds - embargo_delta * (n_folds - 1)
    fold_duration = remaining / n_folds
    if fold_duration <= pd.Timedelta(0):
        raise ValueError("purge/embargo durations leave no evaluation window")
    folds: list[WalkForwardFold] = []
    cursor = initial_end
    for index in range(n_folds):
        purge_start = cursor
        evaluation_start = cursor + purge
        evaluation_end = end if index == n_folds - 1 else evaluation_start + fold_duration
        embargo_end = evaluation_end if index == n_folds - 1 else evaluation_end + embargo_delta
        if evaluation_end <= evaluation_start or embargo_end > end:
            raise ValueError("fold boundaries are invalid")
        folds.append(WalkForwardFold(
            fold_index=index,
            history_start_utc=start.isoformat(),
            history_end_utc=purge_start.isoformat(),
            purge_start_utc=purge_start.isoformat(),
            evaluation_start_utc=evaluation_start.isoformat(),
            evaluation_end_utc=evaluation_end.isoformat(),
            embargo_end_utc=embargo_end.isoformat(),
            history_rows=int((timeline < purge_start).sum()),
            purged_rows=int(((timeline >= purge_start) & (timeline < evaluation_start)).sum()),
            evaluation_rows=int(((timeline >= evaluation_start) & (timeline < evaluation_end)).sum()),
            embargo_rows=int(((timeline >= evaluation_end) & (timeline < embargo_end)).sum()),
            purge_duration=purge.isoformat(), embargo_duration=embargo_delta.isoformat(),
        ))
        cursor = embargo_end
    return tuple(folds)


def moving_block_bootstrap_mean_ci(
    returns: Sequence[float], *, alpha: float = 0.05, block_length: int = 5,
    n_resamples: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Circular moving-block bootstrap CI for mean returns, preserving local order."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("returns must be a nonempty finite one-dimensional sequence")
    if not 0 < alpha < 1 or isinstance(block_length, bool) or not 1 <= block_length <= len(values):
        raise ValueError("alpha or block_length is invalid")
    if isinstance(n_resamples, bool) or n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    estimates = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        starts = rng.integers(0, len(values), size=n_blocks)
        sample = ((starts[:, None] + offsets[None, :]) % len(values)).ravel()[:len(values)]
        estimates[iteration] = values[sample].mean()
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(values.mean()), "lower": float(lower), "upper": float(upper),
        "sample_trades": int(len(values)), "block_length_trades": int(block_length),
        "n_resamples": int(n_resamples), "alpha": float(alpha), "seed": int(seed),
        "method": "circular_moving_block_bootstrap_percentile",
    }


def training_screen(
    fold_results: Mapping[str, Sequence[Mapping[str, Any]]], *, family_size: int,
    minimum_trades: int = 100, block_length: int = 5, n_resamples: int = 2000,
) -> dict[str, Any]:
    """Predeclared conservative gate across all cost scenarios and folds."""
    if family_size < 1 or minimum_trades < 1:
        raise ValueError("family_size and minimum_trades must be positive")
    alpha = 0.05 / family_size
    scenario_scores: dict[str, Any] = {}
    all_pass = bool(fold_results)
    all_reasons: list[str] = []
    for scenario, folds in fold_results.items():
        trade_frames = [fold["trades"] for fold in folds
                        if isinstance(fold.get("trades"), pd.DataFrame)
                        and not fold["trades"].empty]
        trades = (pd.concat(trade_frames, ignore_index=True) if trade_frames
                  else pd.DataFrame(columns=["net_pnl_pips", "entry_timestamp"]))
        if "entry_timestamp" in trades:
            trades = trades.sort_values("entry_timestamp", kind="stable")
        returns = trades["net_pnl_pips"].to_numpy(dtype=float) if not trades.empty else np.array([])
        seed_payload = f"{scenario}|{len(returns)}|{family_size}|{block_length}"
        seed = int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:4], "big")
        ci = None if not len(returns) else moving_block_bootstrap_mean_ci(
            returns, alpha=alpha, block_length=min(block_length, len(returns)),
            n_resamples=n_resamples, seed=seed,
        )
        fold_totals = [float(fold["summary"].get("total_net_pips", 0.0)) for fold in folds]
        reasons: list[str] = []
        if len(returns) < minimum_trades:
            reasons.append("insufficient_training_trades")
        if not fold_totals or any(total <= 0 for total in fold_totals):
            reasons.append("not_positive_in_every_training_fold")
        if ci is None or ci["lower"] <= 0:
            reasons.append("simultaneous_block_bootstrap_lower_bound_not_above_zero")
        scenario_scores[scenario] = {
            "trade_count": int(len(returns)), "fold_net_pips": fold_totals,
            "total_net_pips": float(returns.sum()) if len(returns) else 0.0,
            "mean_net_pips": float(returns.mean()) if len(returns) else None,
            "block_bootstrap_ci": ci, "reasons": reasons,
        }
        all_pass = all_pass and not reasons
        all_reasons.extend(f"{scenario}:{reason}" for reason in reasons)
    return {
        "passed": bool(all_pass), "familywise_alpha": 0.05,
        "family_size_configs": int(family_size), "bonferroni_alpha_per_interval": alpha,
        "minimum_trades": int(minimum_trades), "block_length_trades": int(block_length),
        "scenario_scores": scenario_scores, "reasons": all_reasons,
        "gate_policy": "advance only if each declared cost scenario has at least the minimum trades, positive net P&L in every fold, and a Bonferroni-adjusted moving-block-bootstrap 95%-familywise interval lower bound above zero",
    }


class ResearchEvidenceStore:
    """Additive SQLite ledger for candidate stages, folds, and trade-level evidence."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS research_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    started_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL CHECK(json_valid(request_json)),
                    summary_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(summary_json))
                );
                CREATE TABLE IF NOT EXISTS research_candidate_events (
                    cycle_id TEXT NOT NULL REFERENCES research_cycles(cycle_id),
                    hypothesis_id TEXT NOT NULL,
                    hypothesis_version TEXT NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    details_json TEXT NOT NULL CHECK(json_valid(details_json)),
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY(cycle_id,hypothesis_id,hypothesis_version,sequence_index)
                );
                CREATE TABLE IF NOT EXISTS research_fold_results (
                    cycle_id TEXT NOT NULL REFERENCES research_cycles(cycle_id),
                    hypothesis_id TEXT NOT NULL,
                    hypothesis_version TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    fold_index INTEGER NOT NULL,
                    fold_json TEXT NOT NULL CHECK(json_valid(fold_json)),
                    metrics_json TEXT NOT NULL CHECK(json_valid(metrics_json)),
                    costs_json TEXT NOT NULL CHECK(json_valid(costs_json)),
                    counts_json TEXT NOT NULL CHECK(json_valid(counts_json)),
                    rejection_counts_json TEXT NOT NULL CHECK(json_valid(rejection_counts_json)),
                    uncertainty_json TEXT NOT NULL CHECK(json_valid(uncertainty_json)),
                    PRIMARY KEY(cycle_id,hypothesis_id,hypothesis_version,scenario,fold_index)
                );
                CREATE TABLE IF NOT EXISTS research_trade_evidence (
                    cycle_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    hypothesis_version TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    fold_index INTEGER NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('trade','rejected_signal')),
                    sequence_index INTEGER NOT NULL,
                    record_json TEXT NOT NULL CHECK(json_valid(record_json)),
                    PRIMARY KEY(cycle_id,hypothesis_id,hypothesis_version,scenario,fold_index,event_type,sequence_index),
                    FOREIGN KEY(cycle_id,hypothesis_id,hypothesis_version,scenario,fold_index)
                      REFERENCES research_fold_results(cycle_id,hypothesis_id,hypothesis_version,scenario,fold_index)
                );
            """)

    def start_cycle(self, request: Mapping[str, Any]) -> str:
        cycle_id = str(uuid4())
        with self._connect() as db:
            db.execute("INSERT INTO research_cycles VALUES (?, ?, NULL, 'running', ?, '{}')",
                       (cycle_id, datetime.now(timezone.utc).isoformat(), canonical_json(request)))
        return cycle_id

    def record_candidate_event(
        self, cycle_id: str, hypothesis_id: str, version: str, *, stage: str,
        status: str, reason: str | None = None, details: Mapping[str, Any] | None = None,
    ) -> None:
        with self._connect() as db:
            row = db.execute(
                "SELECT COALESCE(MAX(sequence_index),-1)+1 FROM research_candidate_events WHERE cycle_id=? AND hypothesis_id=? AND hypothesis_version=?",
                (cycle_id, hypothesis_id, version),
            ).fetchone()
            db.execute("INSERT INTO research_candidate_events VALUES (?,?,?,?,?,?,?,?,?)", (
                cycle_id, hypothesis_id, version, int(row[0]), stage, status, reason,
                canonical_json(dict(details or {})), datetime.now(timezone.utc).isoformat(),
            ))

    def record_fold_result(
        self, cycle_id: str, hypothesis_id: str, version: str, scenario: str,
        fold: Mapping[str, Any], *, metrics: Mapping[str, Any], costs: Mapping[str, Any],
        counts: Mapping[str, Any], rejection_counts: Mapping[str, Any],
        uncertainty: Mapping[str, Any] | None = None,
    ) -> None:
        index = int(fold["fold_index"])
        trades = fold.get("trades")
        rejected = fold.get("rejected_signals")
        trade_rows = _record_rows(trades)
        rejected_rows = _record_rows(rejected)
        fold_meta = {key: value for key, value in fold.items()
                     if key not in {"trades", "rejected_signals", "summary", "costs", "counts", "rejected"}}
        with self._connect() as db:
            db.execute("INSERT INTO research_fold_results VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                cycle_id, hypothesis_id, version, scenario, index, canonical_json(fold_meta),
                canonical_json(dict(metrics)), canonical_json(dict(costs)), canonical_json(dict(counts)),
                canonical_json(dict(rejection_counts)), canonical_json(dict(uncertainty or {})),
            ))
            for event_type, rows in (("trade", trade_rows), ("rejected_signal", rejected_rows)):
                db.executemany("INSERT INTO research_trade_evidence VALUES (?,?,?,?,?,?,?,?)", [
                    (cycle_id, hypothesis_id, version, scenario, index, event_type, sequence,
                     canonical_json(row)) for sequence, row in enumerate(rows)
                ])

    def finish_cycle(self, cycle_id: str, status: str, summary: Mapping[str, Any]) -> None:
        if status not in {"completed", "partial_failed", "failed", "interrupted"}:
            raise ValueError("invalid research-cycle status")
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE research_cycles SET status=?, completed_at_utc=?, summary_json=? WHERE cycle_id=? AND status='running'",
                (status, datetime.now(timezone.utc).isoformat(), canonical_json(dict(summary)), cycle_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("research cycle is missing or already completed")

    def get_cycle(self, cycle_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM research_cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
            if row is None:
                raise KeyError(cycle_id)
            result = dict(row)
            result["request"] = json.loads(result.pop("request_json"))
            result["summary"] = json.loads(result.pop("summary_json"))
            result["candidate_events"] = [dict(event) for event in db.execute(
                "SELECT * FROM research_candidate_events WHERE cycle_id=? ORDER BY hypothesis_id,hypothesis_version,sequence_index",
                (cycle_id,),
            )]
            return result


def _record_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        records = value.to_dict(orient="records")
    else:
        records = list(value)
    from .experiment_store import _jsonable
    return [_jsonable(dict(record)) for record in records]
