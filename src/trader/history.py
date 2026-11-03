from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ScanHistory:
    """Small local ledger for scans, predictions and manually recorded outcomes."""

    def __init__(self, path: str | Path = "results/trader_history.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id INTEGER PRIMARY KEY, created_utc TEXT NOT NULL,
                    status TEXT NOT NULL, result_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY, scan_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, model_version TEXT,
                    probability REAL, direction TEXT, created_utc TEXT NOT NULL,
                    candle_time TEXT, features_json TEXT,
                    outcome_return REAL, outcome_recorded_utc TEXT,
                    FOREIGN KEY(scan_id) REFERENCES scans(scan_id)
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_utc DESC);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(predictions)")}
            if "candle_time" not in columns:
                db.execute("ALTER TABLE predictions ADD COLUMN candle_time TEXT")
            if "features_json" not in columns:
                db.execute("ALTER TABLE predictions ADD COLUMN features_json TEXT")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def record_scan(self, result: dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            cur = db.execute("INSERT INTO scans(created_utc,status,result_json) VALUES(?,?,?)",
                             (now, result.get("status", "unknown"), json.dumps(result, allow_nan=False)))
            scan_id = int(cur.lastrowid)
            for item in result.get("observations", []):
                if item.get("probability") is None:
                    continue
                db.execute("""INSERT INTO predictions
                    (scan_id,symbol,timeframe,model_version,probability,direction,created_utc,candle_time,features_json)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (scan_id, item["symbol"], item["timeframe"], item.get("model_version"),
                     item["probability"], item.get("direction"), now, item.get("candle_time"),
                     json.dumps(item.get("features"), allow_nan=False) if item.get("features") else None))
            return scan_id

    def record_outcome(self, prediction_id: int, outcome_return: float) -> None:
        if not isinstance(outcome_return, (int, float)) or not math.isfinite(float(outcome_return)):
            raise ValueError("outcome_return must be a finite number")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            cur = db.execute("UPDATE predictions SET outcome_return=?, outcome_recorded_utc=? "
                             "WHERE prediction_id=? AND outcome_return IS NULL",
                             (float(outcome_return), now, int(prediction_id)))
            if cur.rowcount != 1:
                raise ValueError("prediction was not found or already has an outcome")

    def recent(self, limit: int = 30) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        with self._connect() as db:
            scans = [dict(r) for r in db.execute(
                "SELECT scan_id,created_utc,status,result_json FROM scans ORDER BY scan_id DESC LIMIT ?", (limit,))]
            for row in scans:
                row["result"] = json.loads(row.pop("result_json"))
            predictions = [dict(r) for r in db.execute(
                "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT ?", (limit,))]
            for row in predictions:
                row["features"] = json.loads(row.pop("features_json")) if row.get("features_json") else None
        return {"scans": scans, "predictions": predictions}

    def performance(self) -> dict[str, Any]:
        """Descriptive metrics on explicitly labeled directional predictions."""
        with self._connect() as db:
            rows = [dict(r) for r in db.execute(
                "SELECT probability,outcome_return FROM predictions WHERE outcome_return IS NOT NULL")]
        if not rows:
            return {"labeled_predictions": 0, "directional_accuracy": None,
                    "brier_score": None, "mean_observed_return_pips": None,
                    "disclaimer": "No outcomes recorded; no performance estimate is available."}
        probability = [float(r["probability"]) for r in rows]
        actual = [1.0 if float(r["outcome_return"]) > 0 else 0.0 for r in rows]
        brier = sum((p-y)**2 for p, y in zip(probability, actual)) / len(rows)
        accuracy = sum((p >= .5) == bool(y) for p, y in zip(probability, actual)) / len(rows)
        mean_return = sum(float(r["outcome_return"]) for r in rows) / len(rows)
        return {"labeled_predictions": len(rows), "directional_accuracy": accuracy,
                "brier_score": brier, "mean_observed_return_pips": mean_return,
                "disclaimer": "Descriptive observed outcomes only; costs and selection effects are not adjusted."}

    def labeled_samples(self, *, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        """Return one labeled feature snapshot per candle, oldest first."""
        with self._connect() as db:
            rows = db.execute("""SELECT candle_time,features_json,outcome_return
                FROM predictions WHERE outcome_return IS NOT NULL AND features_json IS NOT NULL
                AND symbol=? AND timeframe=? ORDER BY candle_time,prediction_id""", (symbol, timeframe)).fetchall()
        by_time = {}
        for row in rows:
            features = json.loads(row["features_json"])
            if row["candle_time"] and features:
                by_time[row["candle_time"]] = {**features, "time": row["candle_time"],
                                               "future_pips": float(row["outcome_return"]),
                                               "target": int(float(row["outcome_return"]) > 0)}
        return [by_time[key] for key in sorted(by_time)]
