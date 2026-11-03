"""Synthetic registration, persistence, and orchestration lifecycle tests."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.research.hypothesis_orchestrator as orchestrator
from src.research.hypothesis_orchestrator import prepare_hypothesis_context as build_context
from src.research.experiment_store import ExperimentStore
from src.research.data import validate_m1_cadence
from src.research.hypothesis_runner import run_registered_hypothesis
from src.research.hypothesis_schema import (
    ExecutionAssumptions, ExitRule, FeatureRef, IndicatorInput, RuleNode, StrategyHypothesis,
)


def make_hypothesis(hypothesis_id="synthetic_momentum", version="1.0.0", threshold=0.0):
    return StrategyHypothesis(
        hypothesis_id=hypothesis_id, version=version, strategy_family="momentum",
        rationale="Synthetic test fixture.",
        indicators=(IndicatorInput("return", {"period": 1}),),
        long_entry=RuleNode("gt", (FeatureRef("return_1"), threshold)),
        short_entry=RuleNode("lt", (FeatureRef("return_1"), -threshold if threshold else -0.0)),
        decision_timing="after_signal_candle_close",
        exit_rule=ExitRule("fixed_duration", "5min"), required_warmup_bars=2,
        execution_assumptions=ExecutionAssumptions(transaction_cost_pips=0.25),
    )


def synthetic_market(rows=300):
    x = np.arange(rows)
    close = 1.1 + 0.0004 * np.sin(x / 5) + 0.000001 * x
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC"),
        "open": close, "high": close + 0.0001, "low": close - 0.0001,
        "close": close, "volume": np.ones(rows),
    })


class HypothesisOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "store.sqlite3"
        self.store = ExperimentStore(self.db_path)
        self.market = synthetic_market()
        self.context = orchestrator.prepare_hypothesis_context(self.market)
        self.context_patch = patch.object(orchestrator, "prepare_hypothesis_context",
                                          side_effect=lambda market=None: self.context)
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)

    def registered(self, hypothesis=None):
        hypothesis = hypothesis or make_hypothesis()
        self.store.register_hypothesis(hypothesis)
        return hypothesis

    def test_registration_is_explicit_idempotent_and_versioned(self):
        item = make_hypothesis()
        first = self.store.register_hypothesis(item)
        second = self.store.register_hypothesis(item)
        self.assertEqual(first["semantic_digest"], item.semantic_digest())
        self.assertEqual(second["registered_at_utc"], first["registered_at_utc"])
        v2 = make_hypothesis(version="2.0.0")
        self.store.register_hypothesis(v2)
        self.assertEqual(len(self.store.list_registered_hypotheses()), 2)
        with self.assertRaisesRegex(ValueError, "different content"):
            self.store.register_hypothesis(make_hypothesis(threshold=0.2))
        invalid = StrategyHypothesis(**{**item.__dict__, "decision_timing": "before_close"})
        with self.assertRaises(ValueError):
            self.store.register_hypothesis(invalid)

    def test_additive_legacy_migration_preserves_old_row(self):
        legacy_path = Path(self.temp.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy_path)) as db:
            db.execute("CREATE TABLE experiment_runs (experiment_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL, attempt INTEGER NOT NULL, strategy_id TEXT NOT NULL, strategy_version TEXT NOT NULL, parameters_json TEXT NOT NULL, dataset_fingerprint TEXT NOT NULL, dataset_identity_json TEXT NOT NULL, date_start_utc TEXT NOT NULL, date_end_utc TEXT NOT NULL, execution_assumptions_json TEXT NOT NULL, gap_policy_json TEXT NOT NULL, software_json TEXT NOT NULL, data_quality_json TEXT NOT NULL, started_at_utc TEXT NOT NULL, completed_at_utc TEXT, status TEXT NOT NULL, failure_category TEXT, failure_message TEXT, validation_outcome TEXT)")
            db.execute("INSERT INTO experiment_runs VALUES ('old-id','old-key',1,'legacy','1', '{}', ?, '{}','2025-01-01Z','2025-01-02Z','{}','{}','{}','{}','2025-01-01Z',NULL,'failed','execution_error','legacy failure',NULL)", ("c" * 64,))
            db.commit()
        reopened = ExperimentStore(legacy_path)
        with closing(sqlite3.connect(legacy_path)) as db:
            row = db.execute("SELECT experiment_id,status,failure_message,hypothesis_id,hypothesis_digest,split_metadata_json FROM experiment_runs WHERE experiment_id='old-id'").fetchone()
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(row[:3], ("old-id", "failed", "legacy failure"))
        self.assertIsNone(row[3])
        self.assertIsNone(row[4])
        self.assertEqual(row[5], "{}")
        self.assertIn("registered_hypotheses", tables)

    def test_dry_run_is_read_only_and_rejects_unknown_or_unregistered(self):
        self.registered()
        with closing(sqlite3.connect(self.db_path)) as db:
            before = db.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0]
            orch_before = db.execute("SELECT COUNT(*) FROM orchestration_runs").fetchone()[0]
        with patch.object(orchestrator, "run_registered_hypothesis", side_effect=AssertionError("backtest")):
            preview = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], dry_run=True,
                                                           store_path=self.db_path)
            with self.assertRaises(KeyError):
                orchestrator.run_hypothesis_research(["unknown@1"], dry_run=True,
                                                     store_path=self.db_path)
        self.assertEqual(preview.status, "dry_run")
        self.assertEqual(preview.hypotheses[0]["disposition"], "new")
        with closing(sqlite3.connect(self.db_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0], before)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM orchestration_runs").fetchone()[0], orch_before)
        with self.assertRaises(KeyError):
            orchestrator.run_hypothesis_research(["missing@1"], store_path=self.db_path)

    def test_registered_runner_rejects_unregistered_before_compilation(self):
        with patch("src.research.hypothesis_runner.compile_hypothesis_signals",
                   side_effect=AssertionError("unregistered hypothesis compiled")):
            with self.assertRaises(KeyError):
                run_registered_hypothesis("synthetic_momentum", "1.0.0",
                                          store_path=self.db_path, market_data=self.market)

    def test_exact_dry_run_reuse_and_real_identity_mismatch(self):
        self.registered()
        first = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                       store_path=self.db_path)
        dry = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                    dry_run=True, store_path=self.db_path)
        self.assertEqual(dry.hypotheses[0]["disposition"], "reused_completed")
        self.assertEqual(dry.hypotheses[0]["experiment_id"], first.hypotheses[0]["experiment_id"])
        self.context["software"] = {**self.context["software"], "source_fingerprint_sha256": "f" * 64}
        mismatch = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                         dry_run=True, store_path=self.db_path)
        self.assertEqual(mismatch.hypotheses[0]["disposition"], "new")
        self.assertIn("software_provenance", mismatch.hypotheses[0]["identity_differences"])

    def test_custom_frame_fingerprint_is_stable_and_sensitive(self):
        first, reordered = build_context(self.market), build_context(self.market.iloc[::-1].copy())
        self.assertEqual(first["fingerprint"], reordered["fingerprint"])
        changed = self.market.copy()
        changed.loc[20, ["open", "high", "low", "close"]] += 0.00001
        self.assertNotEqual(first["fingerprint"], build_context(changed)["fingerprint"])
        self.assertEqual(first["identity"]["fingerprint_sha256"], first["fingerprint"])

    def test_m1_declaration_and_cadence_are_enforced_but_market_gaps_are_allowed(self):
        self.registered()
        missing_minute = self.market.drop(index=50).reset_index(drop=True)
        cadence_context = build_context(missing_minute)
        self.assertEqual(cadence_context["quality"]["cadence"]["market_gaps"], 1)
        irregular = self.market.copy()
        irregular.loc[3:, "time"] += pd.Timedelta("30s")
        with self.assertRaisesRegex(ValueError, "M1 timestamps"):
            validate_m1_cadence(irregular["time"])
        two_minute = self.market.copy()
        two_minute["time"] = pd.date_range("2025-01-01", periods=len(two_minute), freq="2min", tz="UTC")
        with self.assertRaisesRegex(ValueError, "M1 cadence"):
            build_context(two_minute)
        exact_ninety = pd.Series(pd.to_datetime([
            "2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z",
            "2025-01-01T00:02:30Z",
        ], utc=True))
        with self.assertRaisesRegex(ValueError, "M1 timestamps|M1 cadence"):
            validate_m1_cadence(exact_ninety)
        with patch.object(orchestrator, "prepare_hypothesis_context", side_effect=build_context):
            with self.assertRaisesRegex(ValueError, "M1 cadence"):
                orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                      market_data=two_minute, store_path=self.db_path)

    def test_unsupported_declared_bar_duration_is_rejected_before_runner(self):
        item = make_hypothesis()
        item = StrategyHypothesis(**{
            **item.__dict__,
            "execution_assumptions": ExecutionAssumptions(
                bar_duration="2min", transaction_cost_pips=0.25,
            ),
        })
        self.registered(item)
        with patch.object(orchestrator, "prepare_hypothesis_context",
                          side_effect=AssertionError("data prepared before declaration rejection")):
            with self.assertRaisesRegex(ValueError, "M1"):
                orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                      dry_run=True, store_path=self.db_path)
        with self.assertRaisesRegex(ValueError, "M1 bar_duration"):
            run_registered_hypothesis("synthetic_momentum", "1.0.0", store=self.store,
                                      market_data=self.market)

    def test_run_persists_identity_metrics_splits_and_never_evaluates_test(self):
        item = self.registered()
        result = orchestrator.run_hypothesis_research([("synthetic_momentum", "1.0.0")],
                                                       store_path=self.db_path)
        self.assertEqual(result.status, "completed")
        row = result.hypotheses[0]
        run = self.store.get_run(row["experiment_id"])
        self.assertEqual(run["hypothesis_id"], item.hypothesis_id)
        self.assertEqual(run["hypothesis_version"], item.version)
        self.assertEqual(run["hypothesis_digest"], item.semantic_digest())
        self.assertEqual(run["parameters"]["canonical_config"], item.canonical_behavior())
        self.assertIn("periods", run["split_metadata"])
        self.assertEqual(self.store.selection_results(row["experiment_id"]).keys(), {"train", "validation"})
        with closing(sqlite3.connect(self.db_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM period_results WHERE experiment_id=? AND period='test'", (row["experiment_id"],)).fetchone()[0], 0)
        with self.assertRaises(KeyError):
            self.store.assert_selection_eligible(row["experiment_id"], "test")
        self.assertTrue(row["test"]["previously_inspected"])
        self.assertFalse(row["test"]["selection_eligible"])

    def test_completed_reuse_avoids_runner_and_different_config_has_new_identity(self):
        self.registered()
        first = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], store_path=self.db_path)
        self.assertEqual(first.lifecycle_counts, {
            "selected": 1, "registered": 1, "executed": 1,
            "completed": 1, "reused": 0, "failed": 0, "skipped": 0,
        })
        with patch.object(orchestrator, "run_registered_hypothesis", side_effect=AssertionError("recomputed")):
            second = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], store_path=self.db_path)
        self.assertEqual(first.hypotheses[0]["experiment_id"], second.hypotheses[0]["experiment_id"])
        self.assertEqual(second.hypotheses[0]["disposition"], "reused_completed")
        self.assertEqual(second.lifecycle_counts["executed"], 0)
        self.assertEqual(second.lifecycle_counts["reused"], 1)
        self.assertEqual(second.lifecycle_counts["completed"], 1)
        self.assertNotIn("generated", second.lifecycle_counts)
        with self.assertRaises(ValueError):
            self.store.create_or_resume_hypothesis_run(
                hypothesis=make_hypothesis(threshold=0.2), dataset_fingerprint="a" * 64,
                dataset_identity={}, date_start_utc=self.context["date_start"],
                date_end_utc=self.context["date_end"], execution_assumptions={}, gap_policy={},
                software=self.context["software"], split_metadata={"periods": {}},
            )

    def test_missing_historical_provenance_or_splits_are_not_reused(self):
        self.registered()
        first = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], store_path=self.db_path)
        experiment_id = first.hypotheses[0]["experiment_id"]
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("UPDATE experiment_runs SET split_metadata_json='{}', software_json='{}' WHERE experiment_id=?",
                       (experiment_id,))
            db.commit()
        dry = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], dry_run=True,
                                                    store_path=self.db_path)
        self.assertEqual(dry.hypotheses[0]["disposition"], "new")
        self.assertIn("split_metadata", dry.hypotheses[0]["identity_differences"])
        second = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], store_path=self.db_path)
        self.assertNotEqual(second.hypotheses[0]["experiment_id"], experiment_id)

    def test_malformed_split_and_identity_fields_block_actual_reuse(self):
        self.registered()
        first = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], store_path=self.db_path)
        first_id = first.hypotheses[0]["experiment_id"]
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("UPDATE experiment_runs SET split_metadata_json=? WHERE experiment_id=?",
                       (json.dumps({"policy": "ordered_complete_candle_timeline"}), first_id))
            db.commit()
        dry = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], dry_run=True,
                                                    store_path=self.db_path)
        self.assertEqual(dry.hypotheses[0]["disposition"], "new")
        second = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], store_path=self.db_path)
        second_id = second.hypotheses[0]["experiment_id"]
        self.assertNotEqual(first_id, second_id)
        with closing(sqlite3.connect(self.db_path)) as db:
            row = db.execute("SELECT dataset_identity_json FROM experiment_runs WHERE experiment_id=?",
                             (second_id,)).fetchone()
            altered_identity = json.loads(row[0])
            altered_identity["row_count"] -= 1
            db.execute("UPDATE experiment_runs SET dataset_identity_json=?, hypothesis_digest=? WHERE experiment_id=?",
                       (json.dumps(altered_identity), "0" * 64, second_id))
            db.commit()
        mismatch = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], dry_run=True,
                                                         store_path=self.db_path)
        self.assertEqual(mismatch.hypotheses[0]["disposition"], "new")
        self.assertIn("hypothesis_digest", mismatch.hypotheses[0]["identity_differences"])
        third = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"], store_path=self.db_path)
        self.assertNotEqual(third.hypotheses[0]["experiment_id"], second_id)
        self.assertEqual(self.store.get_run(first_id)["status"], "completed")
        self.assertEqual(self.store.get_run(second_id)["status"], "completed")

    def test_partial_failure_preserves_attempt_and_retry_creates_distinct_run(self):
        self.registered(make_hypothesis("alpha"))
        self.registered(make_hypothesis("zeta"))
        actual_runner = orchestrator.run_registered_hypothesis

        def fail_alpha(hypothesis_id, version, *, store, market_data):
            if hypothesis_id == "alpha":
                raise RuntimeError("synthetic failure")
            return actual_runner(hypothesis_id, version, store=store, market_data=market_data)

        with patch.object(orchestrator, "run_registered_hypothesis", side_effect=fail_alpha):
            failed = orchestrator.run_hypothesis_research(
                ["zeta@1.0.0", "alpha@1.0.0"], store_path=self.db_path)
        self.assertEqual(failed.status, "partial_failed")
        alpha = next(row for row in failed.hypotheses if row["hypothesis_id"] == "alpha")
        zeta = next(row for row in failed.hypotheses if row["hypothesis_id"] == "zeta")
        failed_id = alpha["experiment_id"]
        self.assertEqual(self.store.get_run(failed_id)["status"], "failed")
        self.assertEqual(self.store.get_run(zeta["experiment_id"])["status"], "completed")
        needs_retry = orchestrator.run_hypothesis_research(["alpha@1.0.0"], store_path=self.db_path)
        self.assertEqual(needs_retry.hypotheses[0]["disposition"], "retry_required")
        retry = orchestrator.run_hypothesis_research(["alpha@1.0.0"], retry_failed=True,
                                                      store_path=self.db_path)
        self.assertNotEqual(retry.hypotheses[0]["experiment_id"], failed_id)
        self.assertEqual(self.store.get_run(failed_id)["status"], "failed")

    def test_interrupted_run_is_resumable_without_overwriting_attempt(self):
        self.registered()
        with patch.object(orchestrator, "run_registered_hypothesis", side_effect=KeyboardInterrupt):
            interrupted = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                                store_path=self.db_path)
        exp_id = interrupted.hypotheses[0]["experiment_id"]
        self.assertEqual(interrupted.status, "interrupted")
        self.assertEqual(self.store.get_run(exp_id)["status"], "running")
        with patch.object(orchestrator, "run_registered_hypothesis", wraps=orchestrator.run_registered_hypothesis) as runner:
            resumed = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                           store_path=self.db_path)
        self.assertEqual(resumed.hypotheses[0]["experiment_id"], exp_id)
        self.assertEqual(resumed.hypotheses[0]["disposition"], "resumed")
        runner.assert_called_once()

    def test_live_execution_claim_blocks_second_orchestrator(self):
        self.registered()
        entered = threading.Event()
        continue_run = threading.Event()
        actual_runner = orchestrator.run_registered_hypothesis

        def held_runner(hypothesis_id, version, *, store, market_data):
            entered.set()
            if not continue_run.wait(timeout=10):
                raise TimeoutError("test execution gate timed out")
            return actual_runner(hypothesis_id, version, store=store, market_data=market_data)

        first_result = []
        with patch.object(orchestrator, "run_registered_hypothesis", side_effect=held_runner):
            worker = threading.Thread(target=lambda: first_result.append(
                orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                     store_path=self.db_path)))
            worker.start()
            self.assertTrue(entered.wait(timeout=5))
            second = orchestrator.run_hypothesis_research(["synthetic_momentum@1.0.0"],
                                                           store_path=self.db_path)
            continue_run.set()
            worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(second.hypotheses[0]["disposition"], "busy")
        self.assertEqual(first_result[0].status, "completed")
        self.assertEqual(self.store.get_run(first_result[0].hypotheses[0]["experiment_id"])["status"], "completed")

    def test_batch_order_and_limit(self):
        self.registered(make_hypothesis("zeta"))
        self.registered(make_hypothesis("alpha"))
        with patch.object(orchestrator, "run_registered_hypothesis", return_value=None):
            # Dry-run exercises requested IDs in stable order without evaluation.
            preview = orchestrator.run_hypothesis_research(["zeta@1.0.0", "alpha@1.0.0"],
                                                           dry_run=True, store_path=self.db_path)
        self.assertEqual([r["hypothesis_id"] for r in preview.hypotheses], ["alpha", "zeta"])
        with self.assertRaisesRegex(ValueError, "maximum"):
            orchestrator.run_hypothesis_research([f"a{i}@1" for i in range(11)], dry_run=True,
                                                 store_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
