"""Synthetic isolated-runner integration tests using the shared backtester."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.hypothesis_schema import (
    ExecutionAssumptions,
    ExitRule,
    FeatureRef,
    IndicatorInput,
    RuleNode,
    StrategyHypothesis,
)
from src.research.experiment_store import ExperimentStore
from src.research.hypothesis_runner import (
    _evaluate_hypothesis_unregistered, run_registered_hypothesis_windows,
)
from src.research.execution_models import BidAskExecutionModel, MarketDataMetadata
from src.research.research_evidence import build_walk_forward_folds


def make_hypothesis(*, holding="2min", delay="0s", cost=1.0):
    return StrategyHypothesis(
        hypothesis_id="runner_integration_fixture",
        version="1.0.0",
        strategy_family="momentum",
        rationale="Synthetic runner integration fixture.",
        indicators=(IndicatorInput("return", {"period": 1}),),
        long_entry=RuleNode("gt", (FeatureRef("return_1"), 0.0)),
        short_entry=RuleNode("lt", (FeatureRef("return_1"), 0.0)),
        decision_timing="after_signal_candle_close",
        exit_rule=ExitRule("fixed_duration", holding),
        required_warmup_bars=2,
        execution_assumptions=ExecutionAssumptions(
            execution_delay=delay, transaction_cost_pips=cost,
        ),
    )


def sinusoidal_market(size=300):
    times = pd.date_range("2025-01-01", periods=size, freq="min", tz="UTC")
    closes = 1.1 + np.sin(np.arange(size) * 0.31) * 0.001
    return pd.DataFrame({
        "time": times,
        "open": closes,
        "high": closes + 0.0001,
        "low": closes - 0.0001,
        "close": closes,
        "volume": np.ones(size),
    })


class HypothesisRunnerTests(unittest.TestCase):
    def test_registered_walk_forward_runner_touches_training_rows_only(self):
        bid = sinusoidal_market(600)
        ask = bid.copy()
        for column in ("open", "high", "low", "close"):
            ask[column] += 0.0002
        # Old validation/test prices are deliberately invalid: the training
        # screen must slice them away before OHLC preparation or feature work.
        bid.loc[360:, "close"] = np.nan
        ask.loc[360:, "close"] = np.nan
        hypothesis = make_hypothesis(cost=0.0)
        metadata_bid = MarketDataMetadata("BID", "EUR/USD", "M1", "bar_open", "UTC", "fixture")
        metadata_ask = MarketDataMetadata("ASK", "EUR/USD", "M1", "bar_open", "UTC", "fixture")
        times = bid["time"]
        outer_start = times.iloc[0]
        outer_end = times.iloc[360]
        folds = build_walk_forward_folds(times, outer_start=outer_start, outer_end=outer_end,
                                         maximum_outcome_horizon="16min", n_folds=2)
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory) / "experiments.sqlite3")
            store.register_hypothesis(hypothesis)
            result = run_registered_hypothesis_windows(
                hypothesis.hypothesis_id, hypothesis.version, folds,
                execution_models=(BidAskExecutionModel(0.0, 0.2), BidAskExecutionModel(0.0, 0.5)),
                store=store, market_data=bid, ask_market_data=ask,
                bid_metadata=metadata_bid, ask_metadata=metadata_ask,
            )
        self.assertEqual(result["outer_training_period"]["rows"], 360)
        self.assertEqual(len(result["folds"]), 2)
        self.assertEqual(len(result["folds"][0]["scenarios"]), 2)
        self.assertEqual(result["validation"]["status"], "not_evaluated_during_training_screen")
        for fold in result["folds"]:
            for period in fold["scenarios"].values():
                self.assertTrue((period["trades"]["exit_timestamp"] <
                                 pd.Timestamp(fold["evaluation_end_utc"])).all())

    def test_registered_evaluator_accepts_explicit_side_aware_model_on_synthetic_data(self):
        bid = sinusoidal_market(150)
        ask = bid.copy()
        for column in ("open", "high", "low", "close"):
            ask[column] += 0.0002
        bid_metadata = MarketDataMetadata("BID", "EUR/USD", "M1", "bar_open", "UTC", "fixture")
        ask_metadata = MarketDataMetadata("ASK", "EUR/USD", "M1", "bar_open", "UTC", "fixture")
        model = BidAskExecutionModel(commission_pips_round_trip=0.2,
                                     slippage_pips_round_trip=0.1)
        result = _evaluate_hypothesis_unregistered(
            make_hypothesis(cost=0.0), bid, execution_model=model, ask_market_data=ask,
            bid_metadata=bid_metadata, ask_metadata=ask_metadata,
        )
        self.assertEqual(result["execution_model"]["identifier"], "bid_ask_side_aware")
        self.assertFalse(result["execution_model"]["legacy_fixed_round_trip_cost_applied"])
        for name in ("train", "validation"):
            trades = result["periods"][name]["trades"]
            long_trades = trades.direction == "long"
            short_trades = trades.direction == "short"
            self.assertTrue((trades.loc[long_trades, "entry_price_side"] == "ASK").all())
            self.assertTrue((trades.loc[long_trades, "exit_price_side"] == "BID").all())
            self.assertTrue((trades.loc[short_trades, "entry_price_side"] == "BID").all())
            self.assertTrue((trades.loc[short_trades, "exit_price_side"] == "ASK").all())
            self.assertTrue((trades.entry_timestamp > trades.decision_timestamp).all())
            np.testing.assert_allclose(trades.spread_cost_pips, 2.0)
            np.testing.assert_allclose(trades.transaction_costs_pips, 0.3)
            np.testing.assert_allclose(trades.net_pnl_pips,
                                       trades.gross_pnl_pips - 0.3)
        self.assertEqual(result["periods"]["test"]["status"], "not_evaluated")

    def test_runner_uses_backtester_timing_cost_and_fixed_holding_duration(self):
        result = _evaluate_hypothesis_unregistered(make_hypothesis(), sinusoidal_market())
        train = result["periods"]["train"]
        validation = result["periods"]["validation"]
        self.assertGreater(train["accepted_trades"], 0)
        self.assertGreater(validation["accepted_trades"], 0)
        for period in (train, validation):
            trades = period["trades"]
            self.assertTrue((trades.entry_timestamp > trades.decision_timestamp).all())
            self.assertTrue(
                ((trades.exit_timestamp - trades.entry_timestamp) == pd.Timedelta("2min")).all()
            )
            np.testing.assert_allclose(
                trades.gross_pnl_pips - trades.net_pnl_pips, 1.0,
            )
            self.assertTrue((trades.exit_timestamp < period["period_end_utc"]).all())
            self.assertEqual(period["accepted_trades"], len(trades))
            self.assertEqual(period["candidate_events"],
                             period["warmup_excluded"] + period["backtester_inputs"])
            self.assertEqual(period["backtester_inputs"],
                             period["accepted_trades"] + period["backtester_rejected"])

    def test_execution_delay_remains_strict_and_is_taken_from_hypothesis(self):
        result = _evaluate_hypothesis_unregistered(make_hypothesis(delay="30s", cost=0.0), sinusoidal_market())
        for name in ("train", "validation"):
            trades = result["periods"][name]["trades"]
            self.assertTrue((trades.entry_timestamp >
                             trades.decision_timestamp + pd.Timedelta("30s")).all())
            self.assertTrue((trades.transaction_costs_pips == 0.0).all())

    def test_outcomes_stay_in_their_period_and_test_is_not_evaluated(self):
        result = _evaluate_hypothesis_unregistered(make_hypothesis(holding="20min"), sinusoidal_market())
        train = result["periods"]["train"]
        validation = result["periods"]["validation"]
        self.assertTrue((train["trades"].exit_timestamp < train["period_end_utc"]).all())
        self.assertTrue((validation["trades"].exit_timestamp < validation["period_end_utc"]).all())
        test = result["periods"]["test"]
        self.assertEqual(test["status"], "not_evaluated")
        self.assertFalse(test["confirmatory"])
        self.assertFalse(test["selection_eligible"])
        self.assertTrue(test["previously_inspected"])
        self.assertIsNone(test["metrics"])

    def test_signal_near_split_boundary_is_rejected_instead_of_crossing(self):
        market = sinusoidal_market(300)
        # Create deterministic directional candidates immediately before the
        # train boundary (row 180); an intended five-minute exit cannot fit.
        market.loc[175, "close"] = 1.1000
        market.loc[176, "close"] = 1.1010
        market.loc[177, "close"] = 1.1020
        market.loc[175:177, "open"] = market.loc[175:177, "close"]
        market.loc[175:177, "high"] = market.loc[175:177, "close"] + 0.0001
        market.loc[175:177, "low"] = market.loc[175:177, "close"] - 0.0001
        result = _evaluate_hypothesis_unregistered(make_hypothesis(holding="5min"), market)
        train = result["periods"]["train"]
        rejected = train["rejected_signals"]
        self.assertGreater(train["backtester_rejected"], 0)
        self.assertTrue(set(rejected["rejection_reason"]).issubset({
            "indicator_warmup_incomplete", "unavailable_entry", "unavailable_exit",
            "gap_exceeds_tolerance", "overlap_skipped",
        }))
        self.assertTrue((train["trades"].exit_timestamp < train["period_end_utc"]).all())

    def test_attribution_is_valid_for_accepted_and_rejected_records(self):
        result = _evaluate_hypothesis_unregistered(make_hypothesis(), sinusoidal_market())
        for name in ("train", "validation"):
            period = result["periods"][name]
            attribution = period["attribution"]
            self.assertEqual(attribution["accepted_verified"], period["accepted_trades"])
            self.assertEqual(attribution["accepted_unavailable"], 0)
            self.assertEqual(attribution["rejected_unavailable"], 0)
            self.assertEqual(
                attribution["rejected_verified"] + attribution["rejected_unavailable"],
                len(period["rejected_signals"]),
            )

    def test_invalid_hypothesis_is_rejected_before_any_market_work(self):
        invalid = make_hypothesis()
        invalid = StrategyHypothesis(
            **{**invalid.__dict__, "decision_timing": "before_close"},
        )
        with self.assertRaisesRegex(ValueError, "after_signal_candle_close"):
            _evaluate_hypothesis_unregistered(invalid, sinusoidal_market())


if __name__ == "__main__":
    unittest.main()
