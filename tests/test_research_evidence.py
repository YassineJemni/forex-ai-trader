"""Synthetic safeguards for purged research evidence and staged evaluation."""

from pathlib import Path
from types import SimpleNamespace
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.execution_models import MarketDataMetadata
from src.research.research_cycle import (
    CostScenario, _side_aware_cost_components, run_evidence_driven_cycle,
)
from src.research.research_evidence import (
    ResearchEvidenceStore, build_walk_forward_folds, moving_block_bootstrap_mean_ci,
    training_screen,
)


def market(rows: int = 1200) -> pd.DataFrame:
    times = pd.date_range("2025-01-06", periods=rows, freq="min", tz="UTC")
    close = 1.1 + np.arange(rows, dtype=float) * 0.000001
    return pd.DataFrame({"time": times, "open": close, "high": close + 0.00002,
                         "low": close - 0.00002, "close": close, "volume": 1.0})


def result_for(fold, *, positive: bool):
    start = pd.Timestamp(fold["evaluation_start_utc"])
    entries = pd.date_range(start, periods=40, freq="min")
    pnl = np.ones(40) if positive else -np.ones(40)
    trades = pd.DataFrame({
        "entry_timestamp": entries,
        "exit_timestamp": entries + pd.Timedelta("15min"),
        "net_pnl_pips": pnl,
        "gross_pnl_pips": pnl,
        "transaction_costs_pips": np.zeros(40),
        "mid_price_pnl_pips": pnl,
        "spread_cost_pips": np.zeros(40),
        "commission_costs_pips": np.zeros(40),
        "slippage_costs_pips": np.zeros(40),
        "signal_id": [f"sig-{fold['fold_index']}-{i}" for i in range(40)],
        "source_row_index": np.arange(40),
        "direction": ["long"] * 40,
    })
    return {
        "summary": {"number_of_trades": 40, "total_net_pips": float(pnl.sum()),
                    "average_net_pips": float(pnl.mean()), "profit_factor": 10.0 if positive else 0.0,
                    "max_drawdown_pips": 0.0 if positive else 40.0},
        "costs": {"gross_total_pips": float(pnl.sum()), "total_assumed_cost_pips": 0.0},
        "trades": trades, "rejected_signals": pd.DataFrame(), "rejected": {},
        "candidate_events": 40, "warmup_excluded": 0, "backtester_inputs": 40,
        "backtester_rejected": 0, "accepted_trades": 40,
    }


class ResearchEvidenceTests(unittest.TestCase):
    def test_side_aware_cost_components_reconcile_without_double_deducting_spread(self):
        trades = pd.DataFrame({
            "gross_pnl_pips": [1.0], "transaction_costs_pips": [0.5],
            "net_pnl_pips": [0.5], "mid_price_pnl_pips": [1.4],
            "spread_cost_pips": [0.4], "commission_costs_pips": [0.0],
            "slippage_costs_pips": [0.5],
        })
        result = _side_aware_cost_components(trades)
        self.assertEqual(result["net_total_pips"], 0.5)
        self.assertEqual(result["spread_deducted_again"], 0.0)
        malformed = trades.copy()
        malformed.loc[0, "net_pnl_pips"] = 0.4
        with self.assertRaises(AssertionError):
            _side_aware_cost_components(malformed)

    def test_fold_windows_are_chronological_and_purged_and_embargoed(self):
        times = pd.Series(pd.date_range("2025-01-01", periods=2000, freq="min", tz="UTC"))
        folds = build_walk_forward_folds(
            times, outer_start=times.iloc[0], outer_end=times.iloc[-1] + pd.Timedelta("1min"),
            maximum_outcome_horizon="16min", n_folds=3, initial_history_fraction=.5,
            embargo="16min",
        )
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(pd.Timestamp(fold.purge_start_utc) - pd.Timestamp(fold.history_end_utc),
                             pd.Timedelta(0))
            self.assertEqual(pd.Timestamp(fold.evaluation_start_utc) - pd.Timestamp(fold.purge_start_utc),
                             pd.Timedelta("16min"))
            self.assertGreater(fold.evaluation_rows, 0)
        for left, right in zip(folds, folds[1:]):
            self.assertEqual(pd.Timestamp(right.purge_start_utc) - pd.Timestamp(left.evaluation_end_utc),
                             pd.Timedelta("16min"))

    def test_fold_builder_rejects_naive_timestamps(self):
        naive = pd.Series(pd.date_range("2025-01-01", periods=100, freq="min"))
        with self.assertRaises(ValueError):
            build_walk_forward_folds(naive, outer_start="2025-01-01T00:00:00Z",
                                     outer_end="2025-01-01T01:40:00Z",
                                     maximum_outcome_horizon="16min", n_folds=2)

    def test_block_bootstrap_is_deterministic_and_screen_is_familywise(self):
        values = np.array([0.2, 0.4, 0.3, 0.5] * 30)
        first = moving_block_bootstrap_mean_ci(values, block_length=5, n_resamples=200, seed=17)
        second = moving_block_bootstrap_mean_ci(values, block_length=5, n_resamples=200, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["method"], "circular_moving_block_bootstrap_percentile")
        screen = training_screen({"scenario": [{
            "summary": {"total_net_pips": 1.0},
            "trades": pd.DataFrame({"net_pnl_pips": values, "entry_timestamp": pd.date_range(
                "2025-01-01", periods=len(values), freq="min", tz="UTC")}),
        }]}, family_size=6, minimum_trades=100, n_resamples=200)
        self.assertEqual(screen["bonferroni_alpha_per_interval"], 0.05 / 6)
        self.assertTrue(screen["passed"])

    def test_empty_training_samples_fail_closed_without_crashing(self):
        screen = training_screen({"scenario": [{
            "summary": {"total_net_pips": 0.0}, "trades": pd.DataFrame(),
        }]}, family_size=1, minimum_trades=100, n_resamples=100)
        self.assertFalse(screen["passed"])
        self.assertIsNone(screen["scenario_scores"]["scenario"]["block_bootstrap_ci"])

    def _cycle(self, root: Path, *, positive: bool):
        bid = market()
        ask = bid.copy()
        ask[["open", "high", "low", "close"]] += 0.00004
        models = (CostScenario("provisional_base", 0.0, 0.5),
                  CostScenario("commission_sensitivity", 0.5, 0.5),
                  CostScenario("higher_slippage_sensitivity", 0.5, 1.0))

        def training_runner(_id, _version, windows, **_kwargs):
            folds = []
            for fold in windows:
                row = {key: getattr(fold, key) for key in fold.__dataclass_fields__}
                row["scenarios"] = {}
                for model in _kwargs["execution_models"]:
                    name = (f"commission_{model.commission_pips_round_trip:g}_"
                            f"slippage_{model.slippage_pips_round_trip:g}")
                    row["scenarios"][name] = result_for(row, positive=positive)
                folds.append(row)
            return {"folds": folds}

        fake_orchestration = SimpleNamespace(hypotheses=[{
            "experiment_id": "synthetic-run", "run_status": "completed", "status": "completed",
            "periods": {"validation": {"metrics": {"number_of_trades": 120},
                                        "counts": {}, "rejection_counts": {}}},
        }])
        metadata = MarketDataMetadata("BID", "EUR/USD", "M1", "bar_open", "UTC", "synthetic")
        ask_metadata = MarketDataMetadata("ASK", "EUR/USD", "M1", "bar_open", "UTC", "synthetic")
        with patch("src.research.research_cycle._training_only_cost_drag_evidence", return_value=[{
            "experiment_id": "prior-train-only", "hypothesis_id": "prior-rsi", "version": "1",
            "semantic_digest": "a" * 64, "trade_count": 500,
            "mid_price_gross_total_pips": 10.0, "net_total_pips": -5.0,
            "assumed_cost_total_pips": 15.0, "used_period": "train",
        }]), patch("src.research.research_cycle.run_registered_hypothesis_windows",
                  side_effect=training_runner), patch(
                      "src.research.research_cycle.run_hypothesis_research",
                      return_value=fake_orchestration) as validation_runner:
            report = run_evidence_driven_cycle(
                store_path=root / "experiments.sqlite3", market_data=bid,
                ask_market_data=ask, cost_scenarios=models,
                bid_metadata=metadata, ask_metadata=ask_metadata,
                bootstrap_resamples=200,
            )
        return report, validation_runner

    def test_failed_training_screen_never_spends_validation_budget_and_persists_trades(self):
        with tempfile.TemporaryDirectory() as directory:
            report, validation_runner = self._cycle(Path(directory), positive=False)
            self.assertFalse(report["training_screen"]["passed"])
            self.assertEqual(validation_runner.call_count, 0)
            self.assertEqual(report["validation_outcomes"], [])
            self.assertIn("test_period", report["protocol"])
            self.assertIn("not evaluated", report["protocol"]["test_period"])
            connection = sqlite3.connect(Path(directory) / "experiments.sqlite3")
            try:
                trade_count = connection.execute(
                    "SELECT count(*) FROM research_trade_evidence WHERE event_type='trade'"
                ).fetchone()[0]
                self.assertGreater(trade_count, 0)
            finally:
                connection.close()

    def test_validation_budget_runs_only_after_training_gate_and_never_test(self):
        with tempfile.TemporaryDirectory() as directory:
            report, validation_runner = self._cycle(Path(directory), positive=True)
            self.assertTrue(report["training_screen"]["passed"])
            self.assertEqual(validation_runner.call_count, 3)  # fixed cost-scenario set, one candidate
            self.assertEqual(len(report["validation_outcomes"]), 3)
            self.assertTrue(all("test" not in item.get("validation", {})
                                for item in report["validation_outcomes"]))
            self.assertEqual(report["protocol"]["validation_budget"], 1)
            self.assertEqual(report["protocol"]["test_period"].split(";")[0], "not evaluated")


if __name__ == "__main__":
    unittest.main()
