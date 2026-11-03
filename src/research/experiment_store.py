"""Local SQLite persistence for reproducible research experiment records."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
from typing import Any, Iterator, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .strategy_registry import StrategySpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "results" / "experiments.sqlite3"
PERIODS = ("train", "validation", "test")
FAILURE_CATEGORIES = {
    "insufficient_data", "invalid_configuration", "execution_error", "persistence_error"
}


def _jsonable(value: Any) -> Any:
    if value is pd.NA or value is None:
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        return stamp.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if value is pd.NaT:
        return None
    return value


def _nullable_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _nullable_int(value: Any) -> int | None:
    if value is None or value == "NaN":
        return None
    return int(value)


def _nullable_float(value: Any) -> float | None:
    if value is None or value == "NaN":
        return None
    return float(value)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-compatible") from exc


_DURATION_FIELDS = {
    "execution_assumptions": frozenset({"bar_duration", "holding_period", "execution_delay"}),
    "gap_policy": frozenset({"tolerance", "indicator_reset_above"}),
}


def _canonical_duration(value: Any, field_name: str) -> str:
    """Normalize a known duration field without interpreting arbitrary strings."""
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a valid duration") from exc
    if pd.isna(duration):
        raise ValueError(f"{field_name} must be a finite duration")
    return duration.isoformat()


def _normalize_duration_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a run request and canonicalize only documented duration keys."""
    normalized = dict(request)
    for mapping_name, field_names in _DURATION_FIELDS.items():
        fields = normalized.get(mapping_name)
        if fields is None:
            continue
        if not isinstance(fields, Mapping):
            raise ValueError(f"{mapping_name} must be a mapping")
        canonical_fields = dict(fields)
        for field_name in field_names:
            if field_name in canonical_fields:
                canonical_fields[field_name] = _canonical_duration(
                    canonical_fields[field_name], f"{mapping_name}.{field_name}",
                )
        normalized[mapping_name] = canonical_fields
    return normalized


def _utc_iso(value: Any, field: str) -> str:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid timezone-aware timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return stamp.tz_convert("UTC").isoformat()


def fingerprint_dataset(path: str | Path) -> str:
    """Return SHA-256 of exact dataset bytes (stable across file paths)."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    digest = hashlib.sha256()
    with dataset_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_metadata(path: str | Path, market: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Content fingerprint plus identifying facts from the loaded data."""
    dataset_path = Path(path).resolve()
    fingerprint = fingerprint_dataset(dataset_path)
    return fingerprint, {
        "filename": dataset_path.name,
        "size_bytes": dataset_path.stat().st_size,
        "row_count": int(len(market)),
        "first_time_utc": market["time"].iloc[0] if len(market) else None,
        "last_time_utc": market["time"].iloc[-1] if len(market) else None,
    }


def fingerprint_market_frame(
    market: pd.DataFrame, *, quality_policy: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Fingerprint the exact normalized frame supplied to a hypothesis runner.

    Rows must already be ordered by unique UTC timestamps, as ensured by the
    hypothesis market-data preparation step. The fingerprint includes column
    order and dtypes, every evaluated row value, and the explicit cleaning/
    cadence policy so that schema or policy changes alter run identity.
    """
    if "time" not in market or not isinstance(market["time"].dtype, pd.DatetimeTZDtype):
        raise ValueError("fingerprinted market data requires timezone-aware time values")
    times = market["time"].dt.tz_convert("UTC")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("fingerprinted market data must have sorted unique valid timestamps")
    schema = [{"name": str(column), "dtype": str(market[column].dtype)} for column in market.columns]
    rows = [[_jsonable(value) for value in row]
            for row in market.itertuples(index=False, name=None)]
    envelope = {
        "fingerprint_scheme": "normalized_market_frame_sha256_v1",
        "quality_policy": dict(quality_policy),
        "schema": schema,
        "rows": rows,
    }
    fingerprint = hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
    identity = {
        "fingerprint_scheme": envelope["fingerprint_scheme"],
        "fingerprint_sha256": fingerprint,
        "quality_policy": dict(quality_policy),
        "columns": schema,
        "row_count": int(len(market)),
        "first_time_utc": times.iloc[0] if len(times) else None,
        "last_time_utc": times.iloc[-1] if len(times) else None,
    }
    return fingerprint, identity


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _validate_hypothesis_identity_metadata(
    *, dataset_fingerprint: str, dataset_identity: Mapping[str, Any],
    date_start_utc: Any, date_end_utc: Any, split_metadata: Mapping[str, Any],
    software: Mapping[str, Any],
) -> None:
    """Require complete frame identity, source provenance, and exact M1 splits."""
    if not _valid_sha256(dataset_fingerprint):
        raise ValueError("hypothesis dataset fingerprint must be a SHA-256 digest")
    if (dataset_identity.get("fingerprint_scheme") != "normalized_market_frame_sha256_v1"
            or dataset_identity.get("fingerprint_sha256") != dataset_fingerprint
            or not isinstance(dataset_identity.get("quality_policy"), Mapping)
            or not isinstance(dataset_identity.get("columns"), list)
            or isinstance(dataset_identity.get("row_count"), bool)
            or not isinstance(dataset_identity.get("row_count"), int)
            or dataset_identity["row_count"] < 3
            or not dataset_identity.get("first_time_utc")
            or not dataset_identity.get("last_time_utc")):
        raise ValueError("hypothesis dataset identity is missing or inconsistent")
    if not _valid_sha256(software.get("source_fingerprint_sha256")):
        raise ValueError("hypothesis run requires a verifiable source fingerprint")
    if not isinstance(split_metadata, Mapping) or set(split_metadata) != {"policy", "fractions", "periods"}:
        raise ValueError("hypothesis split metadata must include policy, fractions, and periods")
    if split_metadata.get("policy") != "ordered_complete_candle_timeline":
        raise ValueError("unsupported hypothesis split policy")
    fractions = split_metadata.get("fractions")
    if not isinstance(fractions, Mapping) or set(fractions) != {"train", "validation", "test"}:
        raise ValueError("split fractions are incomplete")
    try:
        fraction_values = {key: float(fractions[key]) for key in ("train", "validation", "test")}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("split fractions must be numeric") from exc
    if any(not math.isfinite(value) for value in fraction_values.values()) or not (
        math.isclose(fraction_values["train"], 0.6)
        and math.isclose(fraction_values["validation"], 0.2)
        and math.isclose(fraction_values["test"], 0.2)
    ):
        raise ValueError("hypothesis runs require the fixed 60/20/20 split fractions")
    periods = split_metadata.get("periods")
    names = ("train", "validation", "test")
    if not isinstance(periods, Mapping) or set(periods) != set(names):
        raise ValueError("split metadata must contain train, validation, and test boundaries")
    parsed: list[tuple[pd.Timestamp, pd.Timestamp, int, int]] = []
    for name in names:
        item = periods[name]
        if not isinstance(item, Mapping) or set(item) != {
            "start_utc", "end_utc", "start_index", "end_index",
        }:
            raise ValueError(f"{name} split boundary fields are incomplete")
        start, end = pd.Timestamp(item["start_utc"]), pd.Timestamp(item["end_utc"])
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError(f"{name} split boundaries must be ordered timezone-aware timestamps")
        if (isinstance(item["start_index"], bool) or isinstance(item["end_index"], bool)
                or not isinstance(item["start_index"], int) or not isinstance(item["end_index"], int)
                or item["start_index"] < 0 or item["end_index"] <= item["start_index"]):
            raise ValueError(f"{name} split indices must be increasing nonnegative integers")
        parsed.append((start.tz_convert("UTC"), end.tz_convert("UTC"),
                       item["start_index"], item["end_index"]))
    if parsed[0][2] != 0 or parsed[2][3] != dataset_identity["row_count"]:
        raise ValueError("split indices do not cover the complete dataset timeline")
    expected_train_end = int(dataset_identity["row_count"] * 0.6)
    expected_validation_end = int(dataset_identity["row_count"] * 0.8)
    expected_indices = ((0, expected_train_end),
                        (expected_train_end, expected_validation_end),
                        (expected_validation_end, dataset_identity["row_count"]))
    if tuple((item[2], item[3]) for item in parsed) != expected_indices:
        raise ValueError("split indices do not match the fixed chronological 60/20/20 policy")
    for previous, current in zip(parsed, parsed[1:]):
        if previous[1] != current[0] or previous[3] != current[2]:
            raise ValueError("split boundaries must be contiguous and chronological")
    first = pd.Timestamp(dataset_identity["first_time_utc"])
    last = pd.Timestamp(dataset_identity["last_time_utc"])
    request_start, request_end = pd.Timestamp(date_start_utc), pd.Timestamp(date_end_utc)
    if any(stamp.tzinfo is None for stamp in (first, last, request_start, request_end)):
        raise ValueError("dataset and split date identity must be timezone-aware")
    if (parsed[0][0] != first.tz_convert("UTC")
            or parsed[-1][1] != request_end.tz_convert("UTC")
            or request_start.tz_convert("UTC") != first.tz_convert("UTC")
            or parsed[-1][1] != last.tz_convert("UTC") + pd.Timedelta("1min") + pd.Timedelta(1, "ns")):
        raise ValueError("split boundaries do not match the identified dataset date range")


def utc_date_range(market: pd.DataFrame) -> tuple[str, str]:
    """Return an aware interval covering loaded bar timestamps."""
    if market.empty:
        raise ValueError("Cannot define a period for an empty dataset")
    start = pd.Timestamp(market["time"].iloc[0])
    last = pd.Timestamp(market["time"].iloc[-1])
    duration = pd.Timedelta("1min")
    return start.isoformat(), (last + duration + pd.Timedelta(1, "ns")).isoformat()


def code_provenance(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Report Git state when available plus a deterministic source fingerprint."""
    root = Path(project_root).resolve()
    commit: str | None = None
    dirty: bool | None = None
    git_state = "unavailable"
    try:
        git_root = Path(subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()).resolve()
        if git_root != root:
            raise RuntimeError("Git root is not the project root")
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
            raise RuntimeError("Git returned an invalid commit identifier")
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        dirty = bool(status.strip())
        git_state = "dirty" if dirty else "clean"
    except (OSError, RuntimeError, subprocess.SubprocessError):
        pass

    source_root = root / "src"
    digest = hashlib.sha256()
    source_files = sorted(source_root.rglob("*.py")) if source_root.exists() else []
    for path in source_files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "git_commit": commit,
        "git_state": git_state,
        "git_dirty": dirty,
        "source_fingerprint_sha256": digest.hexdigest() if source_files else None,
        "source_files_hashed": len(source_files),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sqlite_version": sqlite3.sqlite_version,
    }


@dataclass(frozen=True)
class RunHandle:
    experiment_id: str
    idempotency_key: str
    attempt: int
    status: str
    resumed: bool = False
    reused_completed: bool = False


class FailedRunRequiresRetry(RuntimeError):
    pass


class ExecutionClaimBusy(RuntimeError):
    """Another live worker currently owns this experiment's execution lease."""


class ExecutionClaimLost(RuntimeError):
    """The worker's lease expired or was claimed by another worker."""


class ExperimentStore:
    """SQLite experiment store; each operation uses a short transaction."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH, *, initialize: bool = True) -> None:
        self.path = Path(path)
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _readonly_connection(self) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    experiment_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    attempt INTEGER NOT NULL CHECK (attempt >= 1),
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    parameters_json TEXT NOT NULL CHECK (json_valid(parameters_json)),
                    dataset_fingerprint TEXT NOT NULL CHECK (length(dataset_fingerprint) = 64),
                    dataset_identity_json TEXT NOT NULL CHECK (json_valid(dataset_identity_json)),
                    date_start_utc TEXT NOT NULL,
                    date_end_utc TEXT NOT NULL,
                    split_metadata_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(split_metadata_json)),
                    execution_assumptions_json TEXT NOT NULL CHECK (json_valid(execution_assumptions_json)),
                    gap_policy_json TEXT NOT NULL CHECK (json_valid(gap_policy_json)),
                    software_json TEXT NOT NULL CHECK (json_valid(software_json)),
                    data_quality_json TEXT NOT NULL CHECK (json_valid(data_quality_json)),
                    started_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                    failure_category TEXT CHECK (failure_category IS NULL OR failure_category IN
                        ('insufficient_data', 'invalid_configuration', 'execution_error', 'persistence_error')),
                    failure_message TEXT,
                    validation_outcome TEXT CHECK (validation_outcome IS NULL OR validation_outcome IN
                        ('negative_net_pips', 'nonnegative_net_pips', 'not_evaluated')),
                    hypothesis_id TEXT,
                    hypothesis_version TEXT,
                    hypothesis_digest TEXT,
                    UNIQUE (idempotency_key, attempt)
                );
                CREATE TABLE IF NOT EXISTS registered_hypotheses (
                    hypothesis_id TEXT NOT NULL,
                    hypothesis_version TEXT NOT NULL,
                    semantic_digest TEXT NOT NULL CHECK (length(semantic_digest) = 64),
                    strategy_family TEXT NOT NULL,
                    canonical_config_json TEXT NOT NULL CHECK (json_valid(canonical_config_json)),
                    hypothesis_json TEXT NOT NULL CHECK (json_valid(hypothesis_json)),
                    validation_status TEXT NOT NULL CHECK (validation_status IN ('valid', 'invalid')),
                    registration_status TEXT NOT NULL CHECK (registration_status IN ('registered')),
                    registered_at_utc TEXT NOT NULL,
                    PRIMARY KEY (hypothesis_id, hypothesis_version)
                );
                CREATE TABLE IF NOT EXISTS period_results (
                    experiment_id TEXT NOT NULL REFERENCES experiment_runs(experiment_id) ON DELETE CASCADE,
                    period TEXT NOT NULL CHECK (period IN ('train', 'validation', 'test')),
                    metrics_json TEXT NOT NULL CHECK (json_valid(metrics_json)),
                    rejection_counts_json TEXT NOT NULL CHECK (json_valid(rejection_counts_json)),
                    costs_json TEXT NOT NULL CHECK (json_valid(costs_json)),
                    data_quality_json TEXT NOT NULL CHECK (json_valid(data_quality_json)),
                    counts_json TEXT NOT NULL CHECK (json_valid(counts_json)),
                    confirmatory INTEGER NOT NULL CHECK (confirmatory IN (0, 1)),
                    selection_eligible INTEGER NOT NULL CHECK (selection_eligible IN (0, 1)),
                    CHECK (period != 'test' OR (confirmatory = 0 AND selection_eligible = 0)),
                    PRIMARY KEY (experiment_id, period)
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_strategy
                    ON experiment_runs(strategy_id, strategy_version, started_at_utc);
                CREATE TABLE IF NOT EXISTS experiment_execution_claims (
                    experiment_id TEXT PRIMARY KEY REFERENCES experiment_runs(experiment_id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL,
                    lease_expires_epoch REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orchestration_runs (
                    orchestration_id TEXT PRIMARY KEY,
                    selected_strategy_ids_json TEXT NOT NULL CHECK (json_valid(selected_strategy_ids_json)),
                    request_json TEXT NOT NULL CHECK (json_valid(request_json)),
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'partial_failed', 'failed', 'interrupted')),
                    started_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    summary_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(summary_json)),
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS orchestration_strategy_runs (
                    orchestration_id TEXT NOT NULL REFERENCES orchestration_runs(orchestration_id) ON DELETE CASCADE,
                    strategy_id TEXT NOT NULL,
                    experiment_id TEXT REFERENCES experiment_runs(experiment_id),
                    disposition TEXT NOT NULL CHECK (disposition IN ('new', 'resumed', 'reused_completed', 'retry_required', 'failed', 'interrupted', 'busy')),
                    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'retry_required', 'busy', 'interrupted', 'not_started')),
                    error_message TEXT,
                    PRIMARY KEY (orchestration_id, strategy_id)
                );
                CREATE TABLE IF NOT EXISTS trade_records (
                    experiment_id TEXT NOT NULL REFERENCES experiment_runs(experiment_id),
                    period TEXT NOT NULL CHECK (period IN ('train','validation','test')),
                    event_type TEXT NOT NULL CHECK (event_type IN ('trade','rejected_signal')),
                    sequence_index INTEGER NOT NULL CHECK (sequence_index >= 0),
                    signal_id TEXT,
                    source_row_index INTEGER,
                    decision_timestamp TEXT,
                    entry_timestamp TEXT,
                    exit_timestamp TEXT,
                    net_pnl_pips REAL,
                    record_json TEXT NOT NULL CHECK (json_valid(record_json)),
                    PRIMARY KEY (experiment_id, period, event_type, sequence_index)
                );
                CREATE INDEX IF NOT EXISTS idx_trade_records_source
                    ON trade_records(experiment_id, period, source_row_index);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(experiment_runs)")}
            if "split_metadata_json" not in columns:
                db.execute("ALTER TABLE experiment_runs ADD COLUMN split_metadata_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(split_metadata_json))")
            for column, declaration in (
                ("hypothesis_id", "TEXT"),
                ("hypothesis_version", "TEXT"),
                ("hypothesis_digest", "TEXT"),
            ):
                if column not in columns:
                    db.execute(f"ALTER TABLE experiment_runs ADD COLUMN {column} {declaration}")

    @staticmethod
    def _request_payload(
        *, strategy: StrategySpec, parameters: Mapping[str, Any], dataset_fingerprint: str,
        dataset_identity: Mapping[str, Any], date_start_utc: Any, date_end_utc: Any,
        execution_assumptions: Mapping[str, Any], gap_policy: Mapping[str, Any],
        software: Mapping[str, Any], data_quality: Mapping[str, Any] | None = None,
        split_metadata: Mapping[str, Any] | None = None,
        hypothesis_identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(dataset_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in dataset_fingerprint.lower()):
            raise ValueError("dataset_fingerprint must be a SHA-256 hex digest")
        start, end = _utc_iso(date_start_utc, "date_start_utc"), _utc_iso(date_end_utc, "date_end_utc")
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError("date range end must be later than start")
        payload = {
            "strategy_id": strategy.strategy_id,
            "strategy_version": strategy.version,
            "parameters": dict(parameters),
            "dataset_fingerprint": dataset_fingerprint.lower(),
            "dataset_identity": dict(dataset_identity),
            "date_start_utc": start,
            "date_end_utc": end,
            "execution_assumptions": dict(execution_assumptions),
            "gap_policy": dict(gap_policy),
            "software": dict(software),
            "data_quality": dict(data_quality or {}),
            "split_metadata": dict(split_metadata or {}),
        }
        if hypothesis_identity is not None:
            payload["hypothesis_identity"] = dict(hypothesis_identity)
        return _normalize_duration_fields(payload)

    @staticmethod
    def _idempotency_key(request: Mapping[str, Any]) -> str:
        return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()

    @staticmethod
    def _row_has_reusable_provenance(row: sqlite3.Row) -> bool:
        """Require stored code provenance and exact split metadata for reuse."""
        if "split_metadata_json" not in row.keys():
            return False
        try:
            split = json.loads(row["split_metadata_json"])
            software = json.loads(row["software_json"])
        except (TypeError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(split, Mapping) and split
            and isinstance(software, Mapping)
            and software.get("source_fingerprint_sha256")
        )

    @staticmethod
    def _row_matches_hypothesis_identity(
        row: sqlite3.Row, identity: Mapping[str, Any] | None,
    ) -> bool:
        if identity is None:
            return True
        required = {
            "hypothesis_id": identity.get("hypothesis_id"),
            "hypothesis_version": identity.get("version"),
            "hypothesis_digest": identity.get("semantic_digest"),
        }
        return all(key in row.keys() and row[key] == value for key, value in required.items())

    @staticmethod
    def _stored_hypothesis_request_matches(row: sqlite3.Row, request: Mapping[str, Any]) -> bool:
        """Compare persisted identity columns and JSON payloads to the full request."""
        identity = request.get("hypothesis_identity")
        if not isinstance(identity, Mapping):
            return False
        try:
            parameters = json.loads(row["parameters_json"])
            dataset_identity = json.loads(row["dataset_identity_json"])
            execution = json.loads(row["execution_assumptions_json"])
            gap = json.loads(row["gap_policy_json"])
            software = json.loads(row["software_json"])
            quality = json.loads(row["data_quality_json"])
            split = json.loads(row["split_metadata_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return False
        try:
            _validate_hypothesis_identity_metadata(
                dataset_fingerprint=row["dataset_fingerprint"],
                dataset_identity=dataset_identity,
                date_start_utc=row["date_start_utc"], date_end_utc=row["date_end_utc"],
                split_metadata=split, software=software,
            )
        except (TypeError, ValueError, KeyError):
            return False
        stored_identity = {
            "hypothesis_id": row["hypothesis_id"],
            "version": row["hypothesis_version"],
            "semantic_digest": row["hypothesis_digest"],
            "canonical_config": parameters.get("canonical_config") if isinstance(parameters, Mapping) else None,
        }
        stored = {
            "strategy_id": row["strategy_id"],
            "strategy_version": row["strategy_version"],
            "parameters": parameters,
            "dataset_fingerprint": row["dataset_fingerprint"],
            "dataset_identity": dataset_identity,
            "date_start_utc": row["date_start_utc"],
            "date_end_utc": row["date_end_utc"],
            "execution_assumptions": execution,
            "gap_policy": gap,
            "software": software,
            "data_quality": quality,
            "split_metadata": split,
            "hypothesis_identity": stored_identity,
        }
        return canonical_json(stored) == canonical_json(request)

    def preview_run(self, **request_args: Any) -> dict[str, Any]:
        """Read-only prediction of create_or_resume_run disposition."""
        retry_failed = bool(request_args.pop("retry_failed", False))
        payload = self._request_payload(**request_args)
        key = self._idempotency_key(payload)
        if not self.path.is_file():
            return {"disposition": "new", "experiment_id": None, "attempt": 1,
                    "status": "not_started", "identity_differences": [],
                    "historical_candidates": []}
        with self._readonly_connection() as db:
            latest = db.execute(
                "SELECT experiment_id, attempt, status FROM experiment_runs "
                "WHERE idempotency_key=? ORDER BY attempt DESC LIMIT 1", (key,),
            ).fetchone()
            historical = db.execute(
                "SELECT * FROM experiment_runs WHERE strategy_id=? "
                "ORDER BY started_at_utc DESC", (payload["strategy_id"],),
            ).fetchall()

        columns = {
            "strategy_version": "strategy_version",
            "parameters": "parameters_json",
            "dataset_fingerprint": "dataset_fingerprint",
            "dataset_identity": "dataset_identity_json",
            "date_start_utc": "date_start_utc",
            "date_end_utc": "date_end_utc",
            "execution_assumptions": "execution_assumptions_json",
            "gap_policy": "gap_policy_json",
            "software_provenance": "software_json",
            "data_quality": "data_quality_json",
            "split_metadata": "split_metadata_json",
        }
        if "hypothesis_identity" in payload:
            columns.update({
                "hypothesis_id": "hypothesis_id",
                "hypothesis_version": "hypothesis_version",
                "hypothesis_digest": "hypothesis_digest",
            })
        requested_fields = {
            "strategy_version": "strategy_version",
            "parameters": "parameters",
            "dataset_fingerprint": "dataset_fingerprint",
            "dataset_identity": "dataset_identity",
            "date_start_utc": "date_start_utc",
            "date_end_utc": "date_end_utc",
            "execution_assumptions": "execution_assumptions",
            "gap_policy": "gap_policy",
            "software_provenance": "software",
            "data_quality": "data_quality",
            "split_metadata": "split_metadata",
        }
        if "hypothesis_identity" in payload:
            requested_fields.update({
                "hypothesis_id": "hypothesis_id",
                "hypothesis_version": "version",
                "hypothesis_digest": "semantic_digest",
            })
        candidate_reports: list[dict[str, Any]] = []
        all_differences: set[str] = set()
        for row in historical:
            differences: list[str] = []
            unrecorded: list[str] = []
            stored_values: dict[str, Any] = {}
            field_differences: dict[str, dict[str, Any]] = {}
            for label, column in columns.items():
                if column not in row.keys():
                    unrecorded.append(label)
                    differences.append(label)
                    field_differences[label] = {
                        "stored": "unrecorded", "requested": (
                            payload["hypothesis_identity"][requested_fields[label]]
                            if label.startswith("hypothesis_")
                            else payload[requested_fields[label]]
                        ),
                    }
                    continue
                value = row[column]
                if label.startswith("hypothesis_") and value is None:
                    unrecorded.append(label)
                    differences.append(label)
                    field_differences[label] = {
                        "stored": "unrecorded",
                        "requested": payload["hypothesis_identity"].get(label.removeprefix("hypothesis_")),
                    }
                    continue
                if column.endswith("_json"):
                    try:
                        value = json.loads(value)
                    except (TypeError, json.JSONDecodeError):
                        unrecorded.append(label)
                        differences.append(label)
                        field_differences[label] = {
                            "stored": "unrecorded or invalid", "requested": (
                                payload["hypothesis_identity"][requested_fields[label]]
                                if label.startswith("hypothesis_")
                                else payload[requested_fields[label]]
                            ),
                        }
                        continue
                if label == "split_metadata" and (not isinstance(value, Mapping) or not value):
                    unrecorded.append(label)
                    differences.append(label)
                    field_differences[label] = {
                        "stored": "unrecorded", "requested": payload[requested_fields[label]],
                    }
                    continue
                if label == "software_provenance":
                    if not isinstance(value, Mapping) or not value.get("source_fingerprint_sha256"):
                        unrecorded.append(label)
                        differences.append(label)
                        field_differences[label] = {
                            "stored": "source fingerprint unrecorded",
                            "requested": payload[requested_fields[label]],
                        }
                        continue
                stored_values[label] = value

            for label in ("hypothesis_id", "hypothesis_version", "hypothesis_digest"):
                if label in stored_values:
                    requested = payload["hypothesis_identity"][requested_fields[label]]
                    if stored_values[label] != requested:
                        differences.append(label)
                        field_differences[label] = {"stored": stored_values[label], "requested": requested}

            # Compare old duration spellings semantically, in memory only. The
            # historical row and its stored idempotency key are never rewritten.
            for label in ("execution_assumptions", "gap_policy"):
                if label in stored_values:
                    try:
                        normalized = _normalize_duration_fields({label: stored_values[label]})
                        stored_values[label] = normalized[label]
                    except ValueError:
                        unrecorded.append(label)
                        differences.append(label)

            for label, value in stored_values.items():
                request_key = requested_fields[label]
                requested = (
                    payload["hypothesis_identity"][request_key]
                    if label.startswith("hypothesis_") else payload[request_key]
                )
                if canonical_json(value) != canonical_json(requested):
                    differences.append(label)
                    field_differences[label] = {
                        "stored": value, "requested": requested,
                    }

            if row["idempotency_key"] == key and not unrecorded and not differences:
                identity_status = "exact_match"
            elif unrecorded:
                identity_status = "unverifiable"
            else:
                identity_status = "mismatch"
                if not differences:
                    differences.append("idempotency_key")
                    field_differences["idempotency_key"] = {
                        "stored": row["idempotency_key"], "requested": key,
                    }
            all_differences.update(differences)
            candidate_reports.append({
                "experiment_id": row["experiment_id"],
                "strategy_id": row["strategy_id"],
                "strategy_version": row["strategy_version"],
                "attempt": int(row["attempt"]),
                "status": row["status"],
                "identity_status": identity_status,
                "safe_to_reuse": identity_status == "exact_match",
                "identity_differences": sorted(set(differences)),
                "field_differences": field_differences,
                "unrecorded_fields": sorted(set(unrecorded)),
            })

        matched_latest = latest
        latest_candidate = next(
            (candidate for candidate in candidate_reports
             if matched_latest is not None
             and candidate["experiment_id"] == matched_latest["experiment_id"]),
            None,
        )
        exact_latest = bool(
            latest is not None and latest_candidate is not None
            and latest_candidate["identity_status"] == "exact_match"
        )
        if not exact_latest:
            latest = None
        differences = [] if exact_latest else sorted(all_differences)
        if latest is None:
            return {"disposition": "new", "experiment_id": None,
                    "attempt": int(matched_latest["attempt"]) + 1 if matched_latest is not None else 1,
                    "status": "not_started", "identity_differences": differences,
                    "historical_candidates": candidate_reports}
        status = latest["status"]
        disposition = (
            "reused_completed" if status == "completed" else
            "resumed" if status == "running" else
            "new_retry" if retry_failed else "retry_required"
        )
        return {
            "disposition": disposition,
            "experiment_id": latest["experiment_id"] if status != "failed" or not retry_failed else None,
            "attempt": int(latest["attempt"]) + (1 if status == "failed" and retry_failed else 0),
            "status": status,
            "identity_differences": [],
            "historical_candidates": candidate_reports,
        }

    def claim_execution(self, experiment_id: str, owner_id: str, *, lease_seconds: float = 900) -> None:
        if not owner_id or lease_seconds <= 0 or not math.isfinite(lease_seconds):
            raise ValueError("owner_id and a finite positive lease_seconds are required")
        now = time.time()
        with self._connection() as db:
            run = db.execute("SELECT status FROM experiment_runs WHERE experiment_id=?", (experiment_id,)).fetchone()
            if run is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            if run["status"] != "running":
                raise ValueError("only a running experiment can be claimed")
            cursor = db.execute("""
                INSERT INTO experiment_execution_claims(experiment_id, owner_id, lease_expires_epoch)
                VALUES (?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    owner_id=excluded.owner_id, lease_expires_epoch=excluded.lease_expires_epoch
                WHERE experiment_execution_claims.lease_expires_epoch <= ?
                   OR experiment_execution_claims.owner_id = excluded.owner_id
            """, (experiment_id, owner_id, now + lease_seconds, now))
            if cursor.rowcount != 1:
                raise ExecutionClaimBusy(f"Experiment {experiment_id} has a live execution claim")

    def renew_execution_claim(self, experiment_id: str, owner_id: str, *, lease_seconds: float = 900) -> bool:
        if lease_seconds <= 0 or not math.isfinite(lease_seconds):
            raise ValueError("lease_seconds must be finite and positive")
        now = time.time()
        with self._connection() as db:
            cursor = db.execute("""
                UPDATE experiment_execution_claims SET lease_expires_epoch=?
                WHERE experiment_id=? AND owner_id=? AND lease_expires_epoch > ?
            """, (now + lease_seconds, experiment_id, owner_id, now))
            return cursor.rowcount == 1

    def release_execution_claim(self, experiment_id: str, owner_id: str) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM experiment_execution_claims WHERE experiment_id=? AND owner_id=?",
                (experiment_id, owner_id),
            )
            return cursor.rowcount == 1

    @contextmanager
    def maintain_execution_claim(
        self, experiment_id: str, owner_id: str, *, lease_seconds: float = 900,
    ) -> Iterator[None]:
        """Claim a run and renew its lease while the caller evaluates it."""
        self.claim_execution(experiment_id, owner_id, lease_seconds=lease_seconds)
        stopped = threading.Event()
        lost = threading.Event()

        def heartbeat() -> None:
            interval = max(0.05, lease_seconds / 3)
            while not stopped.wait(interval):
                try:
                    if not self.renew_execution_claim(
                        experiment_id, owner_id, lease_seconds=lease_seconds,
                    ):
                        lost.set()
                        return
                except sqlite3.Error:
                    lost.set()
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
            if lost.is_set():
                with self._connection() as db:
                    run = db.execute(
                        "SELECT status FROM experiment_runs WHERE experiment_id=?", (experiment_id,),
                    ).fetchone()
                if run is None or run["status"] != "completed":
                    raise ExecutionClaimLost(f"Execution claim for {experiment_id} was lost")
        finally:
            stopped.set()
            thread.join(timeout=max(1.0, lease_seconds / 3 + 0.1))

    def start_orchestration(self, strategy_ids: list[str], request: Mapping[str, Any]) -> str:
        if not strategy_ids:
            raise ValueError("an orchestration requires at least one strategy")
        orchestration_id = str(uuid4())
        with self._connection() as db:
            db.execute("""
                INSERT INTO orchestration_runs (
                    orchestration_id, selected_strategy_ids_json, request_json,
                    status, started_at_utc
                ) VALUES (?, ?, ?, 'running', ?)
            """, (orchestration_id, canonical_json(strategy_ids), canonical_json(dict(request)),
                  datetime.now(timezone.utc).isoformat()))
            for strategy_id in strategy_ids:
                db.execute("""
                    INSERT INTO orchestration_strategy_runs (
                        orchestration_id, strategy_id, disposition, status
                    ) VALUES (?, ?, 'new', 'pending')
                """, (orchestration_id, strategy_id))
        return orchestration_id

    def update_orchestration_strategy(
        self, orchestration_id: str, strategy_id: str, *, disposition: str,
        status: str, experiment_id: str | None = None, error_message: str | None = None,
    ) -> None:
        allowed = {"new", "resumed", "reused_completed", "retry_required", "failed", "interrupted", "busy"}
        if disposition not in allowed:
            raise ValueError(f"disposition must be one of {sorted(allowed)}")
        with self._connection() as db:
            cursor = db.execute("""
                UPDATE orchestration_strategy_runs SET experiment_id=?, disposition=?,
                    status=?, error_message=?
                WHERE orchestration_id=? AND strategy_id=?
            """, (experiment_id, disposition, status, error_message[:4000] if error_message else None,
                  orchestration_id, strategy_id))
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown orchestration strategy {strategy_id!r}")

    def finish_orchestration(
        self, orchestration_id: str, status: str, *, summary: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        if status not in {"completed", "partial_failed", "failed", "interrupted"}:
            raise ValueError("invalid orchestration completion status")
        with self._connection() as db:
            cursor = db.execute("""
                UPDATE orchestration_runs SET status=?, completed_at_utc=?,
                    summary_json=?, error_message=?
                WHERE orchestration_id=? AND status='running'
            """, (status, datetime.now(timezone.utc).isoformat(), canonical_json(dict(summary or {})),
                  error_message[:4000] if error_message else None, orchestration_id))
            if cursor.rowcount != 1:
                raise ValueError("only a running orchestration can be completed")

    def get_orchestration(self, orchestration_id: str) -> dict[str, Any]:
        with self._connection() as db:
            parent = db.execute("SELECT * FROM orchestration_runs WHERE orchestration_id=?", (orchestration_id,)).fetchone()
            if parent is None:
                raise KeyError(f"Unknown orchestration_id: {orchestration_id}")
            children = db.execute(
                "SELECT * FROM orchestration_strategy_runs WHERE orchestration_id=? ORDER BY rowid",
                (orchestration_id,),
            ).fetchall()
        result = dict(parent)
        result["selected_strategy_ids"] = json.loads(result.pop("selected_strategy_ids_json"))
        result["request"] = json.loads(result.pop("request_json"))
        result["summary"] = json.loads(result.pop("summary_json"))
        result["strategies"] = [dict(row) for row in children]
        return result

    def create_or_resume_run(
        self,
        *,
        strategy: StrategySpec,
        parameters: Mapping[str, Any],
        dataset_fingerprint: str,
        dataset_identity: Mapping[str, Any],
        date_start_utc: Any,
        date_end_utc: Any,
        execution_assumptions: Mapping[str, Any],
        gap_policy: Mapping[str, Any],
        software: Mapping[str, Any],
        data_quality: Mapping[str, Any] | None = None,
        split_metadata: Mapping[str, Any] | None = None,
        hypothesis_identity: Mapping[str, Any] | None = None,
        retry_failed: bool = False,
    ) -> RunHandle:
        request = self._request_payload(
            strategy=strategy, parameters=parameters,
            dataset_fingerprint=dataset_fingerprint, dataset_identity=dataset_identity,
            date_start_utc=date_start_utc, date_end_utc=date_end_utc,
            execution_assumptions=execution_assumptions, gap_policy=gap_policy,
            software=software, data_quality=data_quality, split_metadata=split_metadata,
            hypothesis_identity=hypothesis_identity,
        )
        start, end = request["date_start_utc"], request["date_end_utc"]
        key = self._idempotency_key(request)
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            prior = db.execute(
                "SELECT * FROM experiment_runs WHERE idempotency_key=? ORDER BY attempt DESC LIMIT 1",
                (key,),
            ).fetchone()
            latest = prior if (
                prior is not None
                and self._row_has_reusable_provenance(prior)
                and self._row_matches_hypothesis_identity(prior, request.get("hypothesis_identity"))
                and (
                    request.get("hypothesis_identity") is None
                    or self._stored_hypothesis_request_matches(prior, request)
                )
            ) else None
            if latest is not None and latest["status"] == "completed":
                return RunHandle(latest["experiment_id"], key, latest["attempt"], "completed", reused_completed=True)
            if latest is not None and latest["status"] == "running":
                return RunHandle(latest["experiment_id"], key, latest["attempt"], "running", resumed=True)
            if latest is not None and latest["status"] == "failed" and not retry_failed:
                raise FailedRunRequiresRetry(
                    "An identical run failed previously; use retry_failed=True to preserve it and start a new attempt"
                )
            attempt = 1 if prior is None else int(prior["attempt"]) + 1
            experiment_id = str(uuid4())
            db.execute("""
                INSERT INTO experiment_runs (
                    experiment_id, idempotency_key, attempt, strategy_id, strategy_version,
                    parameters_json, dataset_fingerprint, dataset_identity_json,
                    date_start_utc, date_end_utc, execution_assumptions_json,
                    gap_policy_json, software_json, data_quality_json, split_metadata_json,
                    started_at_utc, status, hypothesis_id, hypothesis_version, hypothesis_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
            """, (
                experiment_id, key, attempt, strategy.strategy_id, strategy.version,
                canonical_json(request["parameters"]), request["dataset_fingerprint"],
                canonical_json(request["dataset_identity"]), start, end,
                canonical_json(request["execution_assumptions"]), canonical_json(request["gap_policy"]),
                canonical_json(request["software"]), canonical_json(request["data_quality"]),
                canonical_json(request["split_metadata"]), now,
                (request.get("hypothesis_identity") or {}).get("hypothesis_id"),
                (request.get("hypothesis_identity") or {}).get("version"),
                (request.get("hypothesis_identity") or {}).get("semantic_digest"),
            ))
        return RunHandle(experiment_id, key, attempt, "running")

    def register_hypothesis(self, hypothesis: Any) -> dict[str, Any]:
        """Explicitly register a structurally valid hypothesis.

        Repeating an identical ID/version registration is idempotent. Reusing
        that key for different executable behavior is rejected; distinct
        versions remain separate records.
        """
        from .hypothesis_schema import StrategyHypothesis
        from .hypothesis_validator import validate_hypothesis

        if not isinstance(hypothesis, StrategyHypothesis):
            raise TypeError("hypothesis must be a StrategyHypothesis")
        validation = validate_hypothesis(hypothesis)
        digest = validation.semantic_digest
        config_json = hypothesis.canonical_json()
        full_json = canonical_json(hypothesis.to_dict())
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            prior = db.execute(
                "SELECT * FROM registered_hypotheses WHERE hypothesis_id=? AND hypothesis_version=?",
                (hypothesis.hypothesis_id, hypothesis.version),
            ).fetchone()
            if prior is not None:
                if (prior["semantic_digest"] != digest
                        or prior["canonical_config_json"] != config_json
                        or prior["hypothesis_json"] != full_json):
                    raise ValueError("hypothesis ID/version is already registered with different content")
                return self._registered_hypothesis_dict(prior)
            db.execute("""
                INSERT INTO registered_hypotheses (
                    hypothesis_id, hypothesis_version, semantic_digest, strategy_family,
                    canonical_config_json, hypothesis_json, validation_status,
                    registration_status, registered_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'valid', 'registered', ?)
            """, (hypothesis.hypothesis_id, hypothesis.version, digest,
                  hypothesis.strategy_family, config_json, full_json, timestamp))
            row = db.execute(
                "SELECT * FROM registered_hypotheses WHERE hypothesis_id=? AND hypothesis_version=?",
                (hypothesis.hypothesis_id, hypothesis.version),
            ).fetchone()
        return self._registered_hypothesis_dict(row)

    @staticmethod
    def _registered_hypothesis_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["canonical_config"] = json.loads(result.pop("canonical_config_json"))
        result["hypothesis"] = json.loads(result.pop("hypothesis_json"))
        return result

    def get_registered_hypothesis(self, hypothesis_id: str, version: str) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with self._readonly_connection() as db:
            row = db.execute(
                "SELECT * FROM registered_hypotheses WHERE hypothesis_id=? AND hypothesis_version=?",
                (hypothesis_id, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unregistered hypothesis {hypothesis_id!r} version {version!r}")
        return self._registered_hypothesis_dict(row)

    def list_registered_hypotheses(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self._readonly_connection() as db:
            rows = db.execute(
                "SELECT * FROM registered_hypotheses WHERE registration_status='registered' "
                "ORDER BY hypothesis_id, hypothesis_version"
            ).fetchall()
        return [self._registered_hypothesis_dict(row) for row in rows]

    def create_or_resume_hypothesis_run(
        self, *, hypothesis: Any, dataset_fingerprint: str,
        dataset_identity: Mapping[str, Any], date_start_utc: Any, date_end_utc: Any,
        execution_assumptions: Mapping[str, Any], gap_policy: Mapping[str, Any],
        software: Mapping[str, Any], data_quality: Mapping[str, Any] | None = None,
        split_metadata: Mapping[str, Any] | None = None, retry_failed: bool = False,
    ) -> RunHandle:
        """Create/resume a run only for an explicitly registered hypothesis."""
        from .hypothesis_schema import StrategyHypothesis
        from .hypothesis_validator import validate_hypothesis

        if not isinstance(hypothesis, StrategyHypothesis):
            raise TypeError("hypothesis must be a StrategyHypothesis")
        validation = validate_hypothesis(hypothesis)
        registered = self.get_registered_hypothesis(hypothesis.hypothesis_id, hypothesis.version)
        if (registered["validation_status"] != "valid"
                or registered["registration_status"] != "registered"
                or registered["semantic_digest"] != validation.semantic_digest
                or registered["canonical_config"] != hypothesis.canonical_behavior()):
            raise ValueError("hypothesis registration does not match the supplied validated content")
        _validate_hypothesis_identity_metadata(
            dataset_fingerprint=dataset_fingerprint, dataset_identity=dataset_identity,
            date_start_utc=date_start_utc, date_end_utc=date_end_utc,
            split_metadata=split_metadata or {}, software=software,
        )
        identity = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "version": hypothesis.version,
            "semantic_digest": validation.semantic_digest,
            "canonical_config": hypothesis.canonical_behavior(),
        }
        return self.create_or_resume_run(
            strategy=self._hypothesis_strategy(hypothesis, validation.semantic_digest),
            parameters={"semantic_digest": validation.semantic_digest,
                        "canonical_config": hypothesis.canonical_behavior()},
            dataset_fingerprint=dataset_fingerprint, dataset_identity=dataset_identity,
            date_start_utc=date_start_utc, date_end_utc=date_end_utc,
            execution_assumptions=execution_assumptions, gap_policy=gap_policy,
            software=software, data_quality=data_quality, split_metadata=split_metadata,
            hypothesis_identity=identity, retry_failed=retry_failed,
        )

    @staticmethod
    def _hypothesis_strategy(hypothesis: Any, digest: str) -> StrategySpec:
        return StrategySpec(
            strategy_id=f"hypothesis_{hypothesis.hypothesis_id}",
            name=f"Hypothesis {hypothesis.hypothesis_id} v{hypothesis.version}",
            family=hypothesis.strategy_family,
            hypothesis=hypothesis.rationale,
            rule_description="Registered declarative hypothesis evaluated by hypothesis_runner",
            parameters={"semantic_digest": digest,
                        "canonical_config": hypothesis.canonical_behavior()},
            version=hypothesis.version,
        )

    def preview_hypothesis_run(
        self, *, hypothesis: Any, dataset_fingerprint: str,
        dataset_identity: Mapping[str, Any], date_start_utc: Any, date_end_utc: Any,
        execution_assumptions: Mapping[str, Any], gap_policy: Mapping[str, Any],
        software: Mapping[str, Any], data_quality: Mapping[str, Any] | None = None,
        split_metadata: Mapping[str, Any] | None = None, retry_failed: bool = False,
    ) -> dict[str, Any]:
        """Read-only run preview for a registered hypothesis request."""
        from .hypothesis_schema import StrategyHypothesis
        from .hypothesis_validator import validate_hypothesis

        if not isinstance(hypothesis, StrategyHypothesis):
            raise TypeError("hypothesis must be a StrategyHypothesis")
        validation = validate_hypothesis(hypothesis)
        registered = self.get_registered_hypothesis(hypothesis.hypothesis_id, hypothesis.version)
        if (registered["semantic_digest"] != validation.semantic_digest
                or registered["canonical_config"] != hypothesis.canonical_behavior()):
            raise ValueError("hypothesis is not the exact registered content")
        _validate_hypothesis_identity_metadata(
            dataset_fingerprint=dataset_fingerprint, dataset_identity=dataset_identity,
            date_start_utc=date_start_utc, date_end_utc=date_end_utc,
            split_metadata=split_metadata or {}, software=software,
        )
        return self.preview_run(
            strategy=self._hypothesis_strategy(hypothesis, validation.semantic_digest),
            parameters={"semantic_digest": validation.semantic_digest,
                        "canonical_config": hypothesis.canonical_behavior()},
            dataset_fingerprint=dataset_fingerprint, dataset_identity=dataset_identity,
            date_start_utc=date_start_utc, date_end_utc=date_end_utc,
            execution_assumptions=execution_assumptions, gap_policy=gap_policy,
            software=software, data_quality=data_quality, split_metadata=split_metadata,
            hypothesis_identity={
                "hypothesis_id": hypothesis.hypothesis_id, "version": hypothesis.version,
                "semantic_digest": validation.semantic_digest,
                "canonical_config": hypothesis.canonical_behavior(),
            }, retry_failed=retry_failed,
        )

    @staticmethod
    def _period_payload(result: Mapping[str, Any], period: str) -> dict[str, Any]:
        metrics = result.get("metrics", result.get("summary", {}))
        rejections = result.get("rejection_counts", result.get("rejected", {}))
        costs = dict(result.get("costs", {}))
        trades = result.get("trades")
        if trades is not None and hasattr(trades, "columns"):
            if "transaction_costs_pips" in trades:
                costs.setdefault("total_assumed_cost_pips", float(trades["transaction_costs_pips"].sum()))
            if "gross_pnl_pips" in trades:
                costs.setdefault("gross_total_pips", float(trades["gross_pnl_pips"].sum()))
            for column, key in (
                ("spread_cost_pips", "spread_total_pips_embedded_in_fill_prices"),
                ("commission_costs_pips", "commission_assumption_total_pips"),
                ("slippage_costs_pips", "slippage_assumption_total_pips"),
                ("mid_price_pnl_pips", "mid_price_gross_total_pips"),
            ):
                if column in trades:
                    costs.setdefault(key, float(trades[column].sum()))
        counts = dict(result.get("counts", {}))
        for key in (
            "generated_signals_in_period", "signals_after_warmup", "rejected_warmup",
            "rejected_warmup_initial", "rejected_warmup_after_gap", "rejected_total",
            "accepted_attributed", "rejected_attributed", "rejected_at_period_end",
            "accepted_attribution_unavailable", "rejected_attribution_unavailable",
            "candidate_attribution_verified", "candidate_attribution_unavailable",
        ):
            if key in result:
                counts[key] = result[key]
        quality = result.get("data_quality", result.get("market_data_report", {}))
        # No current experiment has a fresh confirmatory protocol. Train/validation
        # may be used for development; previously accessed test results never may.
        confirmatory = False
        selection_eligible = period in ("train", "validation")
        return {
            "metrics": canonical_json(metrics),
            "rejections": canonical_json(rejections),
            "costs": canonical_json(costs),
            "data_quality": canonical_json(quality),
            "counts": canonical_json(counts),
            "confirmatory": int(confirmatory),
            "selection_eligible": int(selection_eligible),
        }

    def complete_run(
        self,
        experiment_id: str,
        period_results: Mapping[str, Mapping[str, Any]],
        *,
        claim_owner: str | None = None,
    ) -> bool:
        if not {"train", "validation"}.issubset(period_results):
            raise ValueError("completed runs require separate train and validation results")
        if set(period_results) - set(PERIODS):
            raise ValueError("period names must be train, validation, or test")
        payloads = {p: self._period_payload(v, p) for p, v in period_results.items()}
        validation_metrics = json.loads(payloads["validation"]["metrics"])
        total_net = validation_metrics.get("total_net_pips")
        outcome = "not_evaluated" if total_net is None else (
            "negative_net_pips" if float(total_net) < 0 else "nonnegative_net_pips"
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            if claim_owner is not None:
                claim = db.execute(
                    "SELECT owner_id, lease_expires_epoch FROM experiment_execution_claims "
                    "WHERE experiment_id=?", (experiment_id,),
                ).fetchone()
                if claim is None or claim["owner_id"] != claim_owner or claim["lease_expires_epoch"] <= time.time():
                    raise ExecutionClaimLost(f"Execution claim for {experiment_id} was lost")
            run = db.execute(
                "SELECT status FROM experiment_runs WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            if run["status"] == "failed":
                raise ValueError("failed run cannot be completed; create a deliberate retry")
            if run["status"] == "completed":
                existing = db.execute(
                    "SELECT * FROM period_results WHERE experiment_id=? ORDER BY period",
                    (experiment_id,),
                ).fetchall()
                existing_by_period = {row["period"]: row for row in existing}
                for period, payload in payloads.items():
                    row = existing_by_period.get(period)
                    if row is None or any(row[col] != payload[key] for col, key in (
                        ("metrics_json", "metrics"), ("rejection_counts_json", "rejections"),
                        ("costs_json", "costs"), ("data_quality_json", "data_quality"),
                        ("counts_json", "counts"), ("confirmatory", "confirmatory"),
                        ("selection_eligible", "selection_eligible"),
                    )):
                        raise ValueError("completed results differ; refusing to overwrite")
                if set(existing_by_period) != set(payloads):
                    raise ValueError("completed period set differs; refusing to overwrite")
                for period, result in period_results.items():
                    for event_type, field in (("trade", "trades"), ("rejected_signal", "rejected_signals")):
                        rows = self._event_rows(result.get(field))
                        stored = db.execute(
                            "SELECT record_json FROM trade_records WHERE experiment_id=? AND period=? AND event_type=? ORDER BY sequence_index",
                            (experiment_id, period, event_type),
                        ).fetchall()
                        if [row[0] for row in stored] != [canonical_json(row) for row in rows]:
                            raise ValueError("completed trade-level evidence differs; refusing to overwrite")
                return False
            for period, payload in payloads.items():
                db.execute("""
                    INSERT INTO period_results (
                        experiment_id, period, metrics_json, rejection_counts_json,
                        costs_json, data_quality_json, counts_json, confirmatory,
                        selection_eligible
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, period, payload["metrics"], payload["rejections"],
                    payload["costs"], payload["data_quality"], payload["counts"],
                    payload["confirmatory"], payload["selection_eligible"],
                ))
            for period, result in period_results.items():
                for event_type, field in (("trade", "trades"), ("rejected_signal", "rejected_signals")):
                    for sequence_index, record in enumerate(self._event_rows(result.get(field))):
                        normalized = _jsonable(record)
                        db.execute("""
                            INSERT INTO trade_records (
                                experiment_id, period, event_type, sequence_index, signal_id,
                                source_row_index, decision_timestamp, entry_timestamp,
                                exit_timestamp, net_pnl_pips, record_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            experiment_id, period, event_type, sequence_index,
                            _nullable_text(normalized.get("signal_id")),
                            _nullable_int(normalized.get("source_row_index")),
                            normalized.get("decision_timestamp"), normalized.get("entry_timestamp"),
                            normalized.get("exit_timestamp"), _nullable_float(normalized.get("net_pnl_pips")),
                            canonical_json(normalized),
                        ))
            db.execute("""
                UPDATE experiment_runs SET status='completed', completed_at_utc=?,
                    validation_outcome=? WHERE experiment_id=? AND status='running'
            """, (completed_at, outcome, experiment_id))
            if claim_owner is not None:
                db.execute("DELETE FROM experiment_execution_claims WHERE experiment_id=? AND owner_id=?",
                           (experiment_id, claim_owner))
        return True

    @staticmethod
    def _event_rows(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if hasattr(value, "to_dict") and hasattr(value, "columns"):
            return value.to_dict(orient="records")
        if isinstance(value, (list, tuple)):
            return [dict(item) for item in value]
        raise TypeError("trade evidence must be a DataFrame or a sequence of records")

    def get_trade_records(self, experiment_id: str, period: str, *, event_type: str = "trade") -> list[dict[str, Any]]:
        if period not in PERIODS or event_type not in {"trade", "rejected_signal"}:
            raise ValueError("invalid period or trade-record event_type")
        with self._readonly_connection() as db:
            rows = db.execute(
                "SELECT record_json FROM trade_records WHERE experiment_id=? AND period=? AND event_type=? ORDER BY sequence_index",
                (experiment_id, period, event_type),
            ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def record_failure(
        self, experiment_id: str, category: str, message: str, *, claim_owner: str | None = None,
    ) -> None:
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"failure category must be one of {sorted(FAILURE_CATEGORIES)}")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("failure message must be nonempty")
        with self._connection() as db:
            if claim_owner is not None:
                claim = db.execute(
                    "SELECT owner_id FROM experiment_execution_claims WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()
                if claim is None or claim["owner_id"] != claim_owner:
                    raise ExecutionClaimLost(f"Execution claim for {experiment_id} was lost")
            cursor = db.execute("""
                UPDATE experiment_runs SET status='failed', completed_at_utc=?,
                    failure_category=?, failure_message=?
                WHERE experiment_id=? AND status='running'
            """, (datetime.now(timezone.utc).isoformat(), category, message[:4000], experiment_id))
            if cursor.rowcount != 1:
                raise ValueError("only a running experiment can be marked failed")
            if claim_owner is not None:
                db.execute("DELETE FROM experiment_execution_claims WHERE experiment_id=? AND owner_id=?",
                           (experiment_id, claim_owner))

    def get_run(self, experiment_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT * FROM experiment_runs WHERE experiment_id=?", (experiment_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            result = dict(row)
            for key in (
                "parameters_json", "dataset_identity_json", "execution_assumptions_json",
                "gap_policy_json", "software_json", "data_quality_json",
                "split_metadata_json",
            ):
                result[key.removesuffix("_json")] = json.loads(result[key])
            return result

    def get_period_result(self, experiment_id: str, period: str) -> dict[str, Any]:
        if period not in PERIODS:
            raise ValueError(f"period must be one of {PERIODS}")
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM period_results WHERE experiment_id=? AND period=?",
                (experiment_id, period),
            ).fetchone()
        if row is None:
            raise KeyError(f"No {period!r} result for experiment {experiment_id}")
        result = dict(row)
        for key in ("metrics_json", "rejection_counts_json", "costs_json", "data_quality_json", "counts_json"):
            result[key.removesuffix("_json")] = json.loads(result[key])
        return result

    def selection_results(self, experiment_id: str) -> dict[str, dict[str, Any]]:
        """Expose train/validation only; previously inspected test is not selectable."""
        return {period: self.get_period_result(experiment_id, period)
                for period in ("train", "validation")}

    def evaluation_for_report(self, experiment_id: str, period: str) -> dict[str, Any]:
        """Reconstruct the report-facing evaluation from immutable stored fields."""
        stored = self.get_period_result(experiment_id, period)
        counts = stored["counts"]
        return {
            "summary": stored["metrics"],
            "rejected": stored["rejection_counts"],
            "costs": stored["costs"],
            "data_quality": stored["data_quality"],
            **counts,
            "accepted_attributed": counts.get("accepted_attributed", 0),
            "accepted_attribution_unavailable": counts.get("accepted_attribution_unavailable", 0),
            "rejected_attributed": counts.get("rejected_attributed", 0),
            "rejected_attribution_unavailable": counts.get("rejected_attribution_unavailable", 0),
        }

    def assert_selection_eligible(self, experiment_id: str, period: str) -> None:
        result = self.get_period_result(experiment_id, period)
        if not result["selection_eligible"]:
            raise ValueError(f"{period} results are non-confirmatory and ineligible for selection")
