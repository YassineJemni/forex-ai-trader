"""Bounded research-cycle tests use synthetic data and temporary databases."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.experiment_store import ExperimentStore
from src.research.execution_models import MarketDataMetadata
from src.research.hypothesis_templates import TemplateConfig, generate_hypotheses
from src.research.research_cycle import (
    CostScenario, _development_history, analyze_period_results, recommend_next_steps, run_research_cycle,
    select_predeclared_candidates,
)


def synthetic_market(rows: int = 240) -> pd.DataFrame:
    times = pd.date_range("2025-01-06", periods=rows, freq="min", tz="UTC")
    close = 1.1 + np.arange(rows) * 0.000001
    return pd.DataFrame({"time": times, "open": close, "high": close + 0.00002,
                         "low": close - 0.00002, "close": close, "volume": 1.0})


class ResearchCycleTests(unittest.TestCase):
    def test_template_emits_bounded_composite_and_validates(self):
        config = TemplateConfig(
            families=("trend_following",), ema_pairs=((9, 21),),
            include_trend_volatility_filters=True, candidate_limit=2,
        )
        result = generate_hypotheses(config)
        self.assertEqual((result.proposals_generated, result.proposals_rejected,
                          result.proposals_deduplicated), (2, 0, 0))
        self.assertEqual(result.candidate_space_size, 2)
        composite = [h for h in result.hypotheses if len(h.indicators) == 3]
        self.assertEqual(len(composite), 1)
        self.assertEqual(composite[0].required_warmup_bars, 106)

    def test_selection_is_stable_and_does_not_accept_metrics(self):
        proposals = generate_hypotheses(TemplateConfig(
            families=("trend_following",), ema_pairs=((9, 21),),
            include_trend_volatility_filters=True, candidate_limit=2,
        )).hypotheses
        composite = [h for h in proposals if len(h.indicators) == 3]
        first, decisions = select_predeclared_candidates(composite, budget=1)
        second, decisions_again = select_predeclared_candidates(composite, budget=1)
        self.assertEqual(first, second)
        self.assertEqual(decisions, decisions_again)
        self.assertIn("no metrics consulted", decisions[0]["reason"])
        with self.assertRaises(ValueError):
            select_predeclared_candidates(composite, budget=6)

    def test_failure_analysis_uses_train_validation_only_and_labels_small_samples(self):
        periods = {
            "train": {"metrics": {"number_of_trades": 10, "total_net_pips": -3},
                      "costs": {"mid_price_gross_total_pips": -1}},
            "validation": {"metrics": {"number_of_trades": 200, "total_net_pips": 1},
                           "costs": {"mid_price_gross_total_pips": 2}},
            "test": {"metrics": {"number_of_trades": 999, "total_net_pips": 999}},
        }
        result = analyze_period_results(periods, minimum_trades=100)
        self.assertEqual([row["period"] for row in result],
                         ["train", "validation", "train_vs_validation"])
        self.assertTrue(any("trade count below" in note for note in result[0]["notes"]))
        self.assertTrue(any("mid-price gross aggregate is negative" in note
                            for note in result[0]["notes"]))
        self.assertNotIn("test", [row["period"] for row in result])
        recommendations = recommend_next_steps(result)
        self.assertTrue(any("future observations" in item for item in recommendations))
        cost_recommendations = recommend_next_steps([{
            "period": "validation", "notes": [
                "net aggregate is negative after embedded spread and explicit assumed costs",
            ],
        }])
        self.assertTrue(any("commission and execution records" in item
                            for item in cost_recommendations))

    def test_cost_scenario_requires_explicit_valid_nonnegative_assumptions(self):
        with self.assertRaises(ValueError):
            CostScenario("invalid", -0.1, 0.5)
        self.assertEqual(CostScenario("declared", 0.0, 0.5).slippage_pips, 0.5)

    def test_dry_run_does_not_register_or_call_runner(self):
        bid = synthetic_market()
        ask = bid.copy()
        ask[["open", "high", "low", "close"]] += 0.00004
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "experiments.sqlite3"
            store = ExperimentStore(db_path)
            before = len(store.list_registered_hypotheses())
            with patch("src.research.research_cycle.run_hypothesis_research") as runner:
                result = run_research_cycle(
                    store_path=db_path, market_data=bid, ask_market_data=ask,
                    cost_scenarios=(CostScenario("test", 0.0, 0.5),), dry_run=True,
                )
            self.assertEqual(len(store.list_registered_hypotheses()), before)
            self.assertEqual(runner.call_count, 0)
            self.assertEqual(result["cycle_status"], "dry_run")
            self.assertEqual(result["generation"]["proposals_generated"], 2)
            self.assertEqual(result["registrations"][0]["status"],
                             "would_require_explicit_registration")

    def test_synthetic_cycle_uses_official_registered_orchestrator_and_persists(self):
        bid = synthetic_market(360)
        ask = bid.copy()
        ask[["open", "high", "low", "close"]] += 0.00004
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "experiments.sqlite3"
            result = run_research_cycle(
                store_path=db_path, market_data=bid, ask_market_data=ask,
                candidate_budget=1, cost_scenarios=(CostScenario("synthetic", 0.0, 0.5),),
                bid_metadata=MarketDataMetadata("BID", "EUR/USD", "M1", "bar_open", "UTC", "synthetic"),
                ask_metadata=MarketDataMetadata("ASK", "EUR/USD", "M1", "bar_open", "UTC", "synthetic"),
            )
            self.assertEqual(result["lifecycle_counts"]["completed"], 1)
            self.assertEqual(result["lifecycle_counts"]["executed"], 1)
            self.assertEqual(result["executions"][0]["status"], "completed")
            self.assertEqual(set(result["executions"][0]["periods"]), {"train", "validation"})
            self.assertNotIn("test", result["executions"][0]["periods"])
            store = ExperimentStore(db_path, initialize=False)
            registered = store.get_registered_hypothesis(
                result["executions"][0]["hypothesis_id"], result["executions"][0]["version"])
            self.assertEqual(registered["registration_status"], "registered")
            run = store.get_run(result["executions"][0]["experiment_id"])
            self.assertEqual(run["status"], "completed")
            history = _development_history(store)
            self.assertEqual(len(history), 1)
            self.assertEqual(set(history[0]["periods"]), {"train", "validation"})


if __name__ == "__main__":
    unittest.main()
