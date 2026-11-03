"""Synthetic tests for safe hypothesis rule compilation."""

import unittest

import numpy as np
import pandas as pd

from src.research.experiment_ema import validate_signal_attribution
from src.research.hypothesis_schema import (
    ExecutionAssumptions,
    ExitRule,
    FeatureRef,
    IndicatorInput,
    RuleNode,
    StrategyHypothesis,
)
from src.research.hypothesis_signals import compile_hypothesis_signals


def market_from_closes(closes, times=None):
    if times is None:
        times = pd.date_range("2025-01-01", periods=len(closes), freq="min", tz="UTC")
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "time": times,
        "open": close,
        "high": close + 0.0001,
        "low": close - 0.0001,
        "close": close,
        "volume": np.ones(len(close)),
    })


def hypothesis(long_rule, short_rule, *, warmup=3, execution=None):
    return StrategyHypothesis(
        hypothesis_id="synthetic_signal_rule",
        version="1.0.0",
        strategy_family="momentum",
        rationale="Synthetic rule compiler fixture.",
        indicators=(IndicatorInput("return", {"period": 1}),),
        long_entry=long_rule,
        short_entry=short_rule,
        decision_timing="after_signal_candle_close",
        exit_rule=ExitRule("fixed_duration", "2min"),
        required_warmup_bars=warmup,
        execution_assumptions=execution or ExecutionAssumptions(transaction_cost_pips=0.0),
    )


def false_rule():
    return RuleNode("eq", (0, 1))


class HypothesisSignalTests(unittest.TestCase):
    def test_every_supported_operator_and_nested_boolean_semantics(self):
        closes = [1.0, 1.0, 1.1, 1.0, 1.0, 0.9, 1.0]
        market = market_from_closes(closes)
        feature = FeatureRef("return_1")
        comparison_rules = {
            "gt": RuleNode("gt", (feature, 0.0)),
            "gte": RuleNode("gte", (feature, 0.0)),
            "lt": RuleNode("lt", (feature, 0.0)),
            "lte": RuleNode("lte", (feature, 0.0)),
            "eq": RuleNode("eq", (feature, 0.0)),
            "ne": RuleNode("ne", (feature, 0.0)),
            "crosses_above": RuleNode("crosses_above", (feature, 0.0)),
            "crosses_below": RuleNode("crosses_below", (feature, 0.0)),
        }
        expected_signs = {
            "gt": {2, 6}, "gte": {1, 2, 4, 6},
            "lt": {3, 5}, "lte": {1, 3, 4, 5},
            "eq": {1, 4}, "ne": {2, 3, 5, 6},
            "crosses_above": {2, 6}, "crosses_below": {3, 5},
        }
        for operator, rule in comparison_rules.items():
            with self.subTest(operator=operator):
                compiled = compile_hypothesis_signals(
                    hypothesis(rule, false_rule()), market,
                )
                self.assertEqual(set(compiled.signals.source_row_index), expected_signs[operator])

        positive = comparison_rules["gt"]
        nonnegative = comparison_rules["gte"]
        nested = RuleNode("and", (
            RuleNode("or", (positive, RuleNode("not", (comparison_rules["lt"],)))),
            nonnegative,
        ))
        compiled = compile_hypothesis_signals(hypothesis(nested, false_rule()), market)
        self.assertEqual(set(compiled.signals.source_row_index), {1, 2, 4, 6})

    def test_crossing_and_lag_history_never_cross_a_gap_boundary(self):
        start = pd.Timestamp("2025-01-01", tz="UTC")
        times = pd.DatetimeIndex([
            start, start + pd.Timedelta(minutes=1), start + pd.Timedelta(minutes=2),
            start + pd.Timedelta(minutes=3), start + pd.Timedelta(minutes=3, seconds=91),
            start + pd.Timedelta(minutes=4, seconds=91),
        ])
        market = market_from_closes([1.0, 0.9, 0.95, 1.0, 1.1, 1.2], times)
        cross = RuleNode("crosses_above", (FeatureRef("return_1"), 0.0))
        compiled = compile_hypothesis_signals(hypothesis(cross, false_rule()), market)
        # The up-cross in the first segment is valid; the new segment's first
        # return and its first comparison have no prior segment-local state.
        self.assertEqual(compiled.signals.source_row_index.tolist(), [2])

        lagged = RuleNode("and", (
            RuleNode("gt", (FeatureRef("close", 1), FeatureRef("close", 2))),
            RuleNode("gt", (FeatureRef("return_1"), 0.0)),
        ))
        gap_lagged = compile_hypothesis_signals(
            hypothesis(lagged, false_rule(), warmup=3), market,
        )
        self.assertEqual(gap_lagged.signals.source_row_index.tolist(), [3])

        # Use a contiguous segment to check positive offsets denote past bars.
        contiguous = market_from_closes([1.0, 1.1, 1.2, 1.3, 1.4])
        lagged_hypothesis = hypothesis(lagged, false_rule(), warmup=3)
        lagged_signals = compile_hypothesis_signals(lagged_hypothesis, contiguous)
        self.assertEqual(lagged_signals.signals.source_row_index.tolist(), [2, 3, 4])

    def test_warmup_suppresses_candidates_and_unavailable_not_negated_true(self):
        market = market_from_closes([1.0, 0.9, 1.0, 0.9, 1.0])
        rule = RuleNode("not", (RuleNode("gt", (FeatureRef("return_1"), 5.0)),))
        compiled = compile_hypothesis_signals(hypothesis(rule, false_rule(), warmup=4), market)
        self.assertEqual(compiled.candidate_events, 4)
        self.assertEqual(compiled.warmup_suppressed, 2)
        self.assertEqual(compiled.signals.source_row_index.tolist(), [1, 2, 3, 4])
        self.assertEqual(
            compiled.signals.loc[compiled.signals.indicator_warmup_complete,
                                 "source_row_index"].tolist(), [3, 4],
        )

    def test_decisions_follow_candle_close_and_attribution_is_stable(self):
        market = market_from_closes([1.0, 0.9, 1.0, 0.9, 1.0])
        rule = RuleNode("gt", (FeatureRef("return_1"), 0.0))
        compiled = compile_hypothesis_signals(hypothesis(rule, false_rule()), market)
        for signal in compiled.signals.itertuples(index=False):
            source = compiled.market_data.iloc[signal.source_row_index]
            self.assertEqual(signal.source_candle_timestamp, source.time)
            self.assertEqual(signal.decision_timestamp, source.time + pd.Timedelta("1min"))
            self.assertGreater(signal.decision_timestamp, signal.source_candle_timestamp)
        verified, unavailable = validate_signal_attribution(
            compiled.signals, compiled.market_data,
            expected_signal_id_prefix=compiled.signal_id_prefix,
        )
        self.assertEqual((verified, unavailable), (len(compiled.signals), 0))

    def test_future_changes_do_not_change_earlier_signals(self):
        market = market_from_closes([1.0, 1.01, 1.02, 1.0, 0.99, 1.03, 1.04])
        rule = RuleNode("and", (
            RuleNode("gt", (FeatureRef("return_1"), 0.0)),
            RuleNode("gt", (FeatureRef("close", 1), 0.0)),
        ))
        proposal = hypothesis(rule, false_rule())
        baseline = compile_hypothesis_signals(proposal, market).signals
        changed = market.copy()
        changed.loc[5:, ["open", "high", "low", "close"]] += 1.0
        after = compile_hypothesis_signals(proposal, changed).signals
        cutoff = market.loc[4, "time"] + pd.Timedelta("1min")
        left = baseline.loc[baseline.decision_timestamp <= cutoff].reset_index(drop=True)
        right = after.loc[after.decision_timestamp <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_unknown_field_and_invalid_hypothesis_are_rejected_before_compilation(self):
        with self.assertRaisesRegex(ValueError, "unsupported or unavailable"):
            compile_hypothesis_signals(
                hypothesis(RuleNode("gt", (FeatureRef("secret_field"), 0)), false_rule()),
                market_from_closes([1.0, 1.1, 1.2]),
            )
        invalid = hypothesis(RuleNode("python", (1, 2)), false_rule())
        with self.assertRaisesRegex(ValueError, "unsupported rule operator"):
            compile_hypothesis_signals(invalid, market_from_closes([1.0, 1.1, 1.2]))

    def test_raw_price_fields_are_allowed_only_as_completed_bar_values(self):
        rule = RuleNode("gt", (FeatureRef("close"), FeatureRef("high", 1)))
        proposal = hypothesis(
            RuleNode("and", (rule, RuleNode("gt", (FeatureRef("return_1"), 0)))),
            false_rule(), warmup=2,
        )
        market = market_from_closes([1.0, 1.0, 1.1, 1.2])
        compiled = compile_hypothesis_signals(proposal, market)
        self.assertTrue(all(compiled.signals.source_candle_timestamp <=
                            compiled.signals.decision_timestamp))


if __name__ == "__main__":
    unittest.main()
