"""Synthetic orchestration lifecycle and summary tests."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from contextlib import closing
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.research.compare_strategies as compare_module
import src.research.experiment_orchestrator as orchestrator
from src.research.experiment_ema import split_metadata, timeline_periods
from src.research.experiment_store import (
    ExecutionClaimBusy,
    ExperimentStore,
    FailedRunRequiresRetry,
)
from src.research.strategy_registry import get_strategy


class ExperimentOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "results" / "orchestration.sqlite3"
        self.market = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=300, freq="min", tz="UTC"),
            "open": np.full(300, 1.1), "high": np.full(300, 1.1001),
            "low": np.full(300, 1.0999), "close": np.full(300, 1.1),
            "volume": np.ones(300),
        })
        self.market.attrs["load_report"] = {"input_rows": 300, "output_rows": 300, "dropped_rows": 0}
        self.context = {
            "market": self.market,
            "fingerprint": "a" * 64,
            "identity": {"filename": "synthetic.csv", "row_count": 300},
            "date_start": "2025-01-01T00:00:00+00:00",
            "date_end": "2025-01-01T05:00:00.000000001+00:00",
            "quality": dict(self.market.attrs["load_report"]),
            "software": {"git_commit": None, "git_state": "unavailable",
                         "git_dirty": None, "source_fingerprint_sha256": "b" * 64},
            "periods": timeline_periods(self.market),
        }
        self.patches = (
            patch.object(orchestrator, "DEFAULT_DB_PATH", self.db_path),
            patch.object(orchestrator, "prepare_comparison_context", side_effect=lambda: self.context),
        )
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_empty_duplicate_and_unknown_ids_fail_before_data_preparation(self):
        with patch.object(orchestrator, "prepare_comparison_context", side_effect=AssertionError("too late")):
            for ids, message in (([], "at least one"), (["ema_crossover", "ema_crossover"], "duplicate"),
                                 (["not_registered"], "Unregistered")):
                with self.subTest(ids=ids), self.assertRaisesRegex((ValueError, KeyError), message):
                    orchestrator.run_research(ids)
        self.assertFalse(self.db_path.exists())

    def test_dry_run_is_read_only_and_reports_new_disposition(self):
        with patch.object(compare_module, "generate_strategy_signals", side_effect=AssertionError("signals called")):
            result = orchestrator.run_research(["ema_crossover"], dry_run=True)
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.strategies[0]["disposition"], "new")
        self.assertFalse(self.db_path.exists())

    def test_single_and_multiple_runs_preserve_requested_order_and_persist_summary(self):
        result = orchestrator.run_research(
            ["donchian_breakout", "ema_crossover"],
            summary_path=self.root / "summary.md",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.strategy_ids, ["donchian_breakout", "ema_crossover"])
        self.assertEqual([row["strategy_id"] for row in result.strategies], result.strategy_ids)
        self.assertTrue(all(row["disposition"] == "new" for row in result.strategies))
        self.assertTrue((self.root / "summary.md").is_file())
        store = ExperimentStore(self.db_path)
        persisted = store.get_orchestration(result.orchestration_id)
        self.assertEqual(persisted["selected_strategy_ids"], result.strategy_ids)
        for row in result.strategies:
            run = store.get_run(row["experiment_id"])
            self.assertEqual(run["split_metadata"], split_metadata(self.context["periods"]))
            self.assertEqual(row["periods"]["validation"]["metrics"],
                             store.get_period_result(row["experiment_id"], "validation")["metrics"])
            validation = store.get_period_result(row["experiment_id"], "validation")
            self.assertEqual(row["periods"]["validation"]["costs"], validation["costs"])
            self.assertEqual(row["periods"]["validation"]["rejection_counts"],
                             validation["rejection_counts"])
            self.assertEqual(row["periods"]["validation"]["data_quality"],
                             validation["data_quality"])
            test = row["periods"]["test"]
            self.assertTrue(test["previously_inspected"])
            self.assertFalse(test["confirmatory"])
            self.assertFalse(test["selection_eligible"])
            with self.assertRaisesRegex(ValueError, "ineligible"):
                store.assert_selection_eligible(row["experiment_id"], "test")

    def test_completed_run_reused_without_signal_or_backtest_recomputation(self):
        first = orchestrator.run_research(["ema_crossover"])
        with (
            patch.object(compare_module, "generate_strategy_signals", side_effect=AssertionError("signals called")),
            patch.object(compare_module, "evaluate_period", side_effect=AssertionError("backtest called")),
        ):
            second = orchestrator.run_research(["ema_crossover"])
        self.assertEqual(second.strategies[0]["experiment_id"], first.strategies[0]["experiment_id"])
        self.assertEqual(second.strategies[0]["disposition"], "reused_completed")
        with closing(sqlite3.connect(self.db_path)) as db:
            prior_count = db.execute("SELECT COUNT(*) FROM orchestration_runs").fetchone()[0]
        dry = orchestrator.run_research(["ema_crossover"], dry_run=True)
        self.assertEqual(dry.strategies[0]["disposition"], "reused_completed")
        with closing(sqlite3.connect(self.db_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM orchestration_runs").fetchone()[0], prior_count)

    def test_resumes_running_request_and_retries_failed_request_deliberately(self):
        store = ExperimentStore(self.db_path)
        from src.research.compare_strategies import comparison_request_args
        request = comparison_request_args("rsi_mean_reversion", self.context)
        pending = store.create_or_resume_run(**request)
        resumed = orchestrator.run_research(["rsi_mean_reversion"])
        self.assertEqual(resumed.strategies[0]["disposition"], "resumed")
        self.assertEqual(resumed.strategies[0]["experiment_id"], pending.experiment_id)

        retry_request = comparison_request_args("donchian_breakout", self.context)
        failed = store.create_or_resume_run(**retry_request)
        store.record_failure(failed.experiment_id, "execution_error", "synthetic prior failure")
        needs_retry = orchestrator.run_research(["donchian_breakout"])
        self.assertEqual(needs_retry.strategies[0]["disposition"], "retry_required")
        self.assertEqual(needs_retry.strategies[0]["status"], "retry_required")
        retried = orchestrator.run_research(["donchian_breakout"], retry_failed=True)
        self.assertEqual(retried.strategies[0]["disposition"], "new")
        self.assertNotEqual(retried.strategies[0]["experiment_id"], failed.experiment_id)
        self.assertEqual(store.get_run(failed.experiment_id)["status"], "failed")

    def test_execution_claim_excludes_concurrent_owner_and_expiry_allows_recovery(self):
        store = ExperimentStore(self.db_path)
        handle = store.create_or_resume_run(
            strategy=get_strategy("ema_crossover"), parameters={"fast": 9},
            dataset_fingerprint="c" * 64, dataset_identity={"row_count": 1},
            date_start_utc="2025-01-01T00:00:00Z", date_end_utc="2025-01-02T00:00:00Z",
            execution_assumptions={}, gap_policy={}, software={},
        )
        barrier = threading.Barrier(2)
        outcomes = []

        def contender(owner):
            barrier.wait()
            try:
                store.claim_execution(handle.experiment_id, owner, lease_seconds=2)
                outcomes.append((owner, "claimed"))
            except ExecutionClaimBusy:
                outcomes.append((owner, "busy"))

        workers = [threading.Thread(target=contender, args=(owner,))
                   for owner in ("worker-one", "worker-two")]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
        self.assertEqual(sorted(status for _, status in outcomes), ["busy", "claimed"])
        winner = next(owner for owner, status in outcomes if status == "claimed")
        self.assertTrue(store.release_execution_claim(handle.experiment_id, winner))
        store.claim_execution(handle.experiment_id, "worker-one", lease_seconds=0.08)
        time.sleep(0.1)
        store.claim_execution(handle.experiment_id, "worker-two", lease_seconds=1)
        self.assertFalse(store.renew_execution_claim(handle.experiment_id, "worker-one", lease_seconds=1))
        self.assertTrue(store.release_execution_claim(handle.experiment_id, "worker-two"))

    def test_additive_schema_migration_preserves_existing_experiment_results(self):
        store = ExperimentStore(self.db_path)
        handle = store.create_or_resume_run(
            strategy=get_strategy("ema_crossover"), parameters={"fast": 9},
            dataset_fingerprint="e" * 64, dataset_identity={"row_count": 1},
            date_start_utc="2025-01-01T00:00:00Z", date_end_utc="2025-01-02T00:00:00Z",
            execution_assumptions={}, gap_policy={}, software={},
        )
        results = {
            "train": {"summary": {"number_of_trades": 1, "total_net_pips": 2.0}},
            "validation": {"summary": {"number_of_trades": 1, "total_net_pips": -1.0}},
        }
        store.complete_run(handle.experiment_id, results)
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("DROP TABLE orchestration_strategy_runs")
            db.execute("DROP TABLE orchestration_runs")
            db.execute("DROP TABLE experiment_execution_claims")
            db.commit()
        reopened = ExperimentStore(self.db_path)
        self.assertEqual(reopened.get_run(handle.experiment_id)["status"], "completed")
        self.assertEqual(
            reopened.get_period_result(handle.experiment_id, "validation")["metrics"]["total_net_pips"],
            -1.0,
        )
        with closing(sqlite3.connect(self.db_path)) as db:
            names = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        self.assertTrue({"orchestration_runs", "orchestration_strategy_runs",
                         "experiment_execution_claims"}.issubset(names))

    def test_partial_failure_does_not_stop_later_selected_strategy(self):
        original = compare_module.generate_strategy_signals

        def fail_only_rsi(market, *, strategy_names=None):
            if "RSI(14) mean reversion" in (strategy_names or ()):
                raise RuntimeError("synthetic RSI signal failure")
            return original(market, strategy_names=strategy_names)

        with patch.object(compare_module, "generate_strategy_signals", side_effect=fail_only_rsi):
            result = orchestrator.run_research(["rsi_mean_reversion", "donchian_breakout"])
        self.assertEqual(result.status, "partial_failed")
        self.assertEqual([item["status"] for item in result.strategies], ["failed", "completed"])
        self.assertEqual(result.strategies[1]["strategy_id"], "donchian_breakout")

    def test_interruption_is_recorded_and_running_run_remains_resumable(self):
        with patch.object(compare_module, "generate_strategy_signals", side_effect=KeyboardInterrupt):
            result = orchestrator.run_research(["ema_crossover", "rsi_mean_reversion"])
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(result.strategies[0]["status"], "interrupted")
        self.assertEqual(result.strategies[1]["status"], "not_started")
        store = ExperimentStore(self.db_path)
        self.assertEqual(store.get_run(result.strategies[0]["experiment_id"])["status"], "running")
        with closing(sqlite3.connect(self.db_path)) as db:
            claim = db.execute(
                "SELECT COUNT(*) FROM experiment_execution_claims WHERE experiment_id=?",
                (result.strategies[0]["experiment_id"],),
            ).fetchone()[0]
        self.assertEqual(claim, 1)
        self.assertEqual(store.get_orchestration(result.orchestration_id)["status"], "interrupted")

    def test_metadata_drift_is_reported_as_a_new_identity_not_silently_reused(self):
        orchestrator.run_research(["ema_crossover"])
        old_context = self.context
        self.context = {**old_context, "software": {**old_context["software"],
                                                      "source_fingerprint_sha256": "d" * 64}}
        result = orchestrator.run_research(["ema_crossover"], dry_run=True)
        row = result.strategies[0]
        self.assertEqual(row["disposition"], "new")
        self.assertIn("software_provenance", row["identity_differences"])


if __name__ == "__main__":
    unittest.main()
