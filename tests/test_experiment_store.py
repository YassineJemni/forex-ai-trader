"""Lifecycle, idempotency, and persistence tests for SQLite experiment tracking."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.research.compare_strategies as compare_module
import src.research.experiment_ema as ema_module
from src.research.experiment_store import (
    ExperimentStore,
    FailedRunRequiresRetry,
    code_provenance,
    fingerprint_dataset,
)
from src.research.strategy_registry import get_strategy
from src.research.experiment_ema import split_metadata, timeline_periods


class ExperimentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "results" / "experiments.sqlite3"
        self.store = ExperimentStore(self.db_path)
        self.base = {
            "strategy": get_strategy("ema_crossover"),
            "parameters": {"fast": 9, "slow": 21},
            "dataset_fingerprint": "a" * 64,
            "dataset_identity": {"filename": "sample.csv", "row_count": 10},
            "date_start_utc": "2025-01-01T00:00:00Z",
            "date_end_utc": "2025-01-02T00:00:00Z",
            "execution_assumptions": {
                "experiment_kind": "unit_fixture", "cost_pips": 1.0,
                "bar_duration": "1min", "holding_period": "15min", "execution_delay": "0s",
            },
            "gap_policy": {"tolerance": "90s", "indicator_reset_above": "90s"},
            "software": {"git_commit": None, "git_state": "unavailable",
                         "source_fingerprint_sha256": "c" * 64},
            "data_quality": {"dropped_rows": 0},
            "split_metadata": {"policy": "synthetic", "fractions": {"train": 1.0}},
        }

    def create(self, **overrides):
        options = {**self.base, **overrides}
        return self.store.create_or_resume_run(**options)

    @staticmethod
    def results(validation_net=-2.0, include_test=True):
        result = lambda total: {
            "summary": {"number_of_trades": 2, "total_net_pips": total,
                        "average_net_pips": total / 2},
            "rejected": {"gap_exceeds_tolerance": 1, "overlap_skipped": 2},
            "rejected_total": 3,
            "market_data_report": {"rows_used": 10},
            "costs": {"assumed_round_trip_pips": 1.0},
        }
        payload = {"train": result(1.5), "validation": result(validation_net)}
        if include_test:
            payload["test"] = result(3.0)
        return payload

    def test_database_directory_is_created_and_reopening_preserves_results(self):
        self.assertTrue(self.db_path.parent.exists())
        handle = self.create()
        self.store.complete_run(handle.experiment_id, self.results())
        reopened = ExperimentStore(self.db_path)
        run = reopened.get_run(handle.experiment_id)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["validation_outcome"], "negative_net_pips")
        validation = reopened.get_period_result(handle.experiment_id, "validation")
        self.assertEqual(validation["metrics"]["total_net_pips"], -2.0)
        self.assertEqual(validation["rejection_counts"]["overlap_skipped"], 2)

    def test_completed_identical_request_is_reused_without_duplicate_record(self):
        first = self.create()
        self.store.complete_run(first.experiment_id, self.results(include_test=False))
        second = self.create()
        self.assertEqual(second.experiment_id, first.experiment_id)
        self.assertTrue(second.reused_completed)
        with closing(sqlite3.connect(self.db_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0], 1)

    def test_incomplete_run_resumes_with_same_id(self):
        first = self.create()
        resumed = ExperimentStore(self.db_path).create_or_resume_run(**self.base)
        self.assertEqual(first.experiment_id, resumed.experiment_id)
        self.assertTrue(resumed.resumed)
        self.assertEqual(resumed.status, "running")

    def test_failed_attempt_is_preserved_and_retry_gets_new_id(self):
        failed = self.create()
        self.store.record_failure(failed.experiment_id, "execution_error", "synthetic failure")
        with self.assertRaises(FailedRunRequiresRetry):
            self.create()
        retry = self.create(retry_failed=True)
        self.assertNotEqual(failed.experiment_id, retry.experiment_id)
        self.assertEqual(retry.attempt, 2)
        self.assertEqual(self.store.get_run(failed.experiment_id)["status"], "failed")

    def test_train_and_validation_are_required_and_test_is_nonselectable(self):
        handle = self.create()
        with self.assertRaisesRegex(ValueError, "train and validation"):
            self.store.complete_run(handle.experiment_id, {"train": {}})
        self.store.complete_run(handle.experiment_id, self.results())
        self.assertEqual(set(self.store.selection_results(handle.experiment_id)), {"train", "validation"})
        train = self.store.get_period_result(handle.experiment_id, "train")
        validation = self.store.get_period_result(handle.experiment_id, "validation")
        self.assertFalse(train["confirmatory"])
        self.assertFalse(validation["confirmatory"])
        self.assertTrue(train["selection_eligible"])
        self.assertTrue(validation["selection_eligible"])
        test = self.store.get_period_result(handle.experiment_id, "test")
        self.assertFalse(test["confirmatory"])
        self.assertFalse(test["selection_eligible"])
        with self.assertRaisesRegex(ValueError, "ineligible for selection"):
            self.store.assert_selection_eligible(handle.experiment_id, "test")
        self.store.assert_selection_eligible(handle.experiment_id, "validation")

    def test_completed_results_cannot_be_overwritten(self):
        handle = self.create()
        results = self.results(include_test=False)
        self.store.complete_run(handle.experiment_id, results)
        self.assertFalse(self.store.complete_run(handle.experiment_id, results))
        changed = self.results(include_test=False)
        changed["train"]["summary"]["total_net_pips"] = 999
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.store.complete_run(handle.experiment_id, changed)

    def test_trade_and_rejection_evidence_persists_with_source_attribution(self):
        handle = self.create()
        results = self.results(include_test=False)
        trade = {
            "signal_id": "signal-7", "source_row_index": 7,
            "source_candle_timestamp": pd.Timestamp("2025-01-01T00:07:00Z"),
            "decision_timestamp": pd.Timestamp("2025-01-01T00:08:00Z"),
            "direction": "long", "entry_timestamp": pd.Timestamp("2025-01-01T00:09:00Z"),
            "exit_timestamp": pd.Timestamp("2025-01-01T00:24:00Z"),
            "entry_price": 1.1, "exit_price": 1.1002,
            "gross_pnl_pips": 2.0, "spread_cost_pips": 0.4,
            "commission_costs_pips": 0.0, "slippage_costs_pips": 0.5,
            "net_pnl_pips": 1.1,
        }
        rejected = {"signal_id": "signal-8", "source_row_index": 8,
                    "decision_timestamp": pd.Timestamp("2025-01-01T00:10:00Z"),
                    "rejection_reason": "gap_exceeds_tolerance"}
        for period in ("train", "validation"):
            results[period]["trades"] = pd.DataFrame([trade])
            results[period]["rejected_signals"] = pd.DataFrame([rejected])
        self.store.complete_run(handle.experiment_id, results)
        persisted = self.store.get_trade_records(handle.experiment_id, "train")
        self.assertEqual(persisted[0]["signal_id"], "signal-7")
        self.assertEqual(persisted[0]["source_row_index"], 7)
        self.assertEqual(persisted[0]["entry_timestamp"], "2025-01-01T00:09:00+00:00")
        rejects = self.store.get_trade_records(handle.experiment_id, "train", event_type="rejected_signal")
        self.assertEqual(rejects[0]["rejection_reason"], "gap_exceeds_tolerance")
        self.assertFalse(self.store.complete_run(handle.experiment_id, results))
        results["train"]["trades"].loc[0, "net_pnl_pips"] = 999
        with self.assertRaisesRegex(ValueError, "trade-level evidence differs"):
            self.store.complete_run(handle.experiment_id, results)

    def test_invalid_metadata_and_failure_categories_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.create(dataset_fingerprint="bad")
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.create(date_start_utc="2025-01-01")
        handle = self.create()
        with self.assertRaisesRegex(ValueError, "failure category"):
            self.store.record_failure(handle.experiment_id, "unknown", "bad")
        self.store.record_failure(handle.experiment_id, "insufficient_data", "too few rows")
        self.assertEqual(self.store.get_run(handle.experiment_id)["failure_category"], "insufficient_data")

    def test_dataset_fingerprint_tracks_content_and_not_path(self):
        left, right = self.root / "a.csv", self.root / "b.csv"
        left.write_bytes(b"time,open\n1,1.1\n")
        right.write_bytes(left.read_bytes())
        self.assertEqual(fingerprint_dataset(left), fingerprint_dataset(right))
        right.write_bytes(b"time,open\n1,1.2\n")
        self.assertNotEqual(fingerprint_dataset(left), fingerprint_dataset(right))

    def test_code_provenance_does_not_invent_git_metadata(self):
        info = code_provenance(self.root)
        self.assertEqual(info["git_state"], "unavailable")
        self.assertIsNone(info["git_commit"])
        self.assertIsNone(info["git_dirty"])
        self.assertTrue(info["python_version"])
        self.assertTrue(info["pandas_version"])
        self.assertTrue(info["numpy_version"])

    def test_database_constraints_disallow_test_selection(self):
        handle = self.create()
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store._connection() as db:
                db.execute("""
                    INSERT INTO period_results VALUES (?, 'test', '{}', '{}', '{}', '{}', '{}', 1, 1)
                """, (handle.experiment_id,))

    def test_known_duration_fields_are_canonicalized_for_identity_only(self):
        from src.research.experiment_store import ExperimentStore

        original = ExperimentStore._request_payload(**self.base)
        equivalent = {
            **self.base,
            "execution_assumptions": {
                **self.base["execution_assumptions"],
                "bar_duration": "60 seconds",
                "holding_period": "0 days 00:15:00",
                "execution_delay": "0 days 00:00:00",
                "experiment_kind": "unit_fixture",
            },
            "gap_policy": {
                "tolerance": "0 days 00:01:30",
                "indicator_reset_above": "90 seconds",
            },
        }
        canonical = ExperimentStore._request_payload(**equivalent)
        self.assertEqual(ExperimentStore._idempotency_key(original),
                         ExperimentStore._idempotency_key(canonical))
        self.assertEqual(canonical["execution_assumptions"]["experiment_kind"], "unit_fixture")

        different = {
            **equivalent,
            "execution_assumptions": {
                **equivalent["execution_assumptions"], "holding_period": "16min",
            },
        }
        self.assertNotEqual(
            ExperimentStore._idempotency_key(original),
            ExperimentStore._idempotency_key(ExperimentStore._request_payload(**different)),
        )
        different_kind = {
            **self.base,
            "execution_assumptions": {
                **self.base["execution_assumptions"], "experiment_kind": "other_kind",
            },
        }
        self.assertNotEqual(
            ExperimentStore._idempotency_key(original),
            ExperimentStore._idempotency_key(ExperimentStore._request_payload(**different_kind)),
        )

    def test_preview_reports_all_candidates_and_is_read_only(self):
        first = self.create()
        self.store.record_failure(first.experiment_id, "execution_error", "synthetic")
        second = self.create(parameters={"fast": 10, "slow": 21})
        before = self.store.get_run(second.experiment_id)
        requested = dict(self.base)
        # Recreate the original request as an older completed record.
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("UPDATE experiment_runs SET status='completed' WHERE experiment_id=?", (first.experiment_id,))
            db.commit()
        with closing(sqlite3.connect(self.db_path)) as db:
            snapshot_before = db.execute(
                "SELECT experiment_id, idempotency_key, attempt, status, split_metadata_json "
                "FROM experiment_runs ORDER BY experiment_id"
            ).fetchall()
        request = self.base
        preview = self.store.preview_run(**request)
        self.assertEqual({candidate["experiment_id"] for candidate in preview["historical_candidates"]},
                         {first.experiment_id, second.experiment_id})
        by_id = {candidate["experiment_id"]: candidate for candidate in preview["historical_candidates"]}
        self.assertEqual(by_id[first.experiment_id]["identity_status"], "exact_match")
        self.assertEqual(by_id[second.experiment_id]["identity_status"], "mismatch")
        self.assertEqual(by_id[first.experiment_id]["status"], "completed")
        self.assertTrue(by_id[first.experiment_id]["safe_to_reuse"])
        self.assertIn("parameters", by_id[second.experiment_id]["field_differences"])
        after = self.store.get_run(second.experiment_id)
        self.assertEqual(before, after)
        with closing(sqlite3.connect(self.db_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0], 2)
            snapshot_after = db.execute(
                "SELECT experiment_id, idempotency_key, attempt, status, split_metadata_json "
                "FROM experiment_runs ORDER BY experiment_id"
            ).fetchall()
        self.assertEqual(snapshot_after, snapshot_before)

    def test_missing_historical_split_metadata_is_unverifiable_and_not_reused(self):
        handle = self.create()
        self.store.complete_run(handle.experiment_id, self.results(include_test=False))
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("UPDATE experiment_runs SET split_metadata_json='{}' WHERE experiment_id=?",
                       (handle.experiment_id,))
            db.commit()
        old_key = self.store.get_run(handle.experiment_id)["idempotency_key"]
        preview = self.store.preview_run(**self.base)
        self.assertEqual(preview["disposition"], "new")
        candidate = preview["historical_candidates"][0]
        self.assertEqual(candidate["identity_status"], "unverifiable")
        self.assertIn("split_metadata", candidate["unrecorded_fields"])
        self.assertEqual(candidate["field_differences"]["split_metadata"]["stored"], "unrecorded")
        self.assertFalse(candidate["safe_to_reuse"])
        retry = self.create()
        self.assertNotEqual(retry.experiment_id, handle.experiment_id)
        self.assertEqual(self.store.get_run(handle.experiment_id)["idempotency_key"], old_key)
        self.assertEqual(self.store.get_run(handle.experiment_id)["split_metadata"], {})

    def test_existing_runners_persist_without_changing_their_fixed_rules(self):
        market = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=300, freq="min", tz="UTC"),
            "open": np.full(300, 1.1), "high": np.full(300, 1.1001),
            "low": np.full(300, 1.0999), "close": np.full(300, 1.1),
            "volume": np.ones(300),
        })
        market.attrs["load_report"] = {"input_rows": 300, "output_rows": 300}
        dataset_path = self.root / "dataset.csv"
        dataset_path.write_text("synthetic fixture fingerprint source", encoding="utf-8")
        store = self.store
        provenance = {"git_commit": None, "git_state": "unavailable",
                      "git_dirty": None, "source_fingerprint_sha256": "b" * 64}

        with (
            patch.object(compare_module, "load_market_data", return_value=market),
            patch.object(compare_module, "DEFAULT_DATA_PATH", dataset_path),
            patch.object(compare_module, "ExperimentStore", return_value=store),
            patch.object(compare_module, "code_provenance", return_value=provenance),
        ):
            compare_module.run_comparison(self.root / "comparison.md")

            expected_compare_split = split_metadata(timeline_periods(market))

        with (
            patch.object(ema_module, "load_market_data", return_value=market),
            patch.object(ema_module, "DEFAULT_DATA_PATH", dataset_path),
            patch.object(ema_module, "ExperimentStore", return_value=store),
            patch.object(ema_module, "code_provenance", return_value=provenance),
        ):
            ema_module.run_experiment(report_path=self.root / "ema.md")

            expected_ema_split = split_metadata(timeline_periods(market))

        # Completed requests are reportable from SQLite without signal or trade recomputation.
        with (
            patch.object(compare_module, "load_market_data", return_value=market),
            patch.object(compare_module, "DEFAULT_DATA_PATH", dataset_path),
            patch.object(compare_module, "ExperimentStore", return_value=store),
            patch.object(compare_module, "code_provenance", return_value=provenance),
            patch.object(compare_module, "generate_strategy_signals", side_effect=AssertionError("must reuse")),
            patch.object(compare_module, "evaluate_period", side_effect=AssertionError("must reuse")),
        ):
            compare_module.run_comparison(self.root / "comparison-reused.md")

        with (
            patch.object(ema_module, "load_market_data", return_value=market),
            patch.object(ema_module, "DEFAULT_DATA_PATH", dataset_path),
            patch.object(ema_module, "ExperimentStore", return_value=store),
            patch.object(ema_module, "code_provenance", return_value=provenance),
            patch.object(ema_module, "generate_ema_crossover_signals", side_effect=AssertionError("must reuse")),
            patch.object(ema_module, "evaluate_period", side_effect=AssertionError("must reuse")),
        ):
            ema_module.run_experiment(report_path=self.root / "ema-reused.md")

        with closing(sqlite3.connect(self.db_path)) as db:
            rows = db.execute(
                "SELECT strategy_id, status FROM experiment_runs ORDER BY strategy_id"
            ).fetchall()
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(status == "completed" for _, status in rows))
            periods = db.execute(
                "SELECT period, confirmatory, selection_eligible FROM period_results "
                "WHERE experiment_id=(SELECT experiment_id FROM experiment_runs "
                "WHERE strategy_id='rsi_mean_reversion') ORDER BY period"
            ).fetchall()
            self.assertEqual(periods, [("test", 0, 0), ("train", 0, 1), ("validation", 0, 1)])
        with closing(sqlite3.connect(self.db_path)) as db:
            stored_splits = dict(db.execute(
                "SELECT strategy_id, split_metadata_json FROM experiment_runs"
            ).fetchall())
        self.assertEqual(json.loads(stored_splits["ema_crossover"]), expected_ema_split)
        self.assertEqual(json.loads(stored_splits["rsi_mean_reversion"]), expected_compare_split)
        for report_name in ("comparison-reused.md", "ema-reused.md"):
            self.assertTrue((self.root / report_name).is_file())

    def test_comparison_startup_failure_leaves_earlier_run_resumable(self):
        market = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=300, freq="min", tz="UTC"),
            "open": np.full(300, 1.1), "high": np.full(300, 1.1001),
            "low": np.full(300, 1.0999), "close": np.full(300, 1.1),
            "volume": np.ones(300),
        })
        market.attrs["load_report"] = {"input_rows": 300, "output_rows": 300}
        dataset_path = self.root / "startup.csv"
        dataset_path.write_text("synthetic startup fixture", encoding="utf-8")
        provenance = {"git_commit": None, "git_state": "unavailable", "git_dirty": None,
                      "source_fingerprint_sha256": "d" * 64}
        real_create = self.store.create_or_resume_run
        calls = 0

        def fail_second_request(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FailedRunRequiresRetry("synthetic later startup failure")
            return real_create(**kwargs)

        with (
            patch.object(compare_module, "load_market_data", return_value=market),
            patch.object(compare_module, "DEFAULT_DATA_PATH", dataset_path),
            patch.object(compare_module, "ExperimentStore", return_value=self.store),
            patch.object(compare_module, "code_provenance", return_value=provenance),
            patch.object(self.store, "create_or_resume_run", side_effect=fail_second_request),
        ):
            with self.assertRaises(FailedRunRequiresRetry):
                compare_module.run_comparison(self.root / "never-written.md")

        with closing(sqlite3.connect(self.db_path)) as db:
            rows = db.execute("SELECT strategy_id, status FROM experiment_runs").fetchall()
        self.assertEqual(rows, [("ema_crossover", "running")])

    def test_comparison_reuses_completed_strategy_independently(self):
        market = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=300, freq="min", tz="UTC"),
            "open": np.full(300, 1.1), "high": np.full(300, 1.1001),
            "low": np.full(300, 1.0999), "close": np.full(300, 1.1),
            "volume": np.ones(300),
        })
        market.attrs["load_report"] = {"input_rows": 300, "output_rows": 300}
        dataset_path = self.root / "partial-reuse.csv"
        dataset_path.write_text("synthetic partial-reuse fixture", encoding="utf-8")
        provenance = {"git_commit": None, "git_state": "unavailable", "git_dirty": None,
                      "source_fingerprint_sha256": "e" * 64}
        with (
            patch.object(compare_module, "load_market_data", return_value=market),
            patch.object(compare_module, "DEFAULT_DATA_PATH", dataset_path),
            patch.object(compare_module, "ExperimentStore", return_value=self.store),
            patch.object(compare_module, "code_provenance", return_value=provenance),
        ):
            compare_module.run_comparison(self.root / "initial.md")

        with closing(sqlite3.connect(self.db_path)) as db:
            ids = [row[0] for row in db.execute(
                "SELECT experiment_id FROM experiment_runs WHERE strategy_id IN "
                "('rsi_mean_reversion', 'donchian_breakout')"
            )]
            db.executemany("DELETE FROM period_results WHERE experiment_id=?", [(value,) for value in ids])
            db.executemany("DELETE FROM experiment_runs WHERE experiment_id=?", [(value,) for value in ids])
            db.commit()

        original_signals = compare_module.generate_strategy_signals
        original_evaluate = compare_module.evaluate_period
        with (
            patch.object(compare_module, "load_market_data", return_value=market),
            patch.object(compare_module, "DEFAULT_DATA_PATH", dataset_path),
            patch.object(compare_module, "ExperimentStore", return_value=self.store),
            patch.object(compare_module, "code_provenance", return_value=provenance),
            patch.object(compare_module, "generate_strategy_signals", wraps=original_signals) as signals_spy,
            patch.object(compare_module, "evaluate_period", wraps=original_evaluate) as evaluate_spy,
        ):
            compare_module.run_comparison(self.root / "partial-reuse.md")

        self.assertEqual(
            signals_spy.call_args.kwargs["strategy_names"],
            ["RSI(14) mean reversion", "Donchian 20 breakout"],
        )
        self.assertEqual(evaluate_spy.call_count, 6)


if __name__ == "__main__":
    unittest.main()
