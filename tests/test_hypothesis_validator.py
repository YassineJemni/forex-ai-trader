"""Structural safety and current-pipeline support tests for hypotheses."""

import unittest

from dataclasses import replace

from src.research.hypothesis_schema import (
    ExecutionAssumptions,
    ExitRule,
    FeatureRef,
    IndicatorInput,
    RuleNode,
)
from src.research.hypothesis_validator import (
    COMPARISON_OPERATORS,
    INDICATORS,
    SUPPORTED_OPERATORS,
    validate_hypothesis,
    validate_unique_hypotheses,
)
from tests.test_hypothesis_schema import make_hypothesis


class HypothesisValidatorTests(unittest.TestCase):
    def test_all_declared_indicator_outputs_match_current_feature_columns(self):
        expected = {
            "ema": ("ema_9", "ema_21", "ema_50"),
            "ema_gap": ("ema_gap",),
            "ema_long_gap": ("ema_long_gap",),
            "return": ("return_1", "return_5", "return_15", "return_60"),
            "rsi": ("rsi_14",),
            "volatility": ("volatility_15", "volatility_60"),
            "candle_range": ("candle_range",),
            "candle_body": ("candle_body",),
            "time_of_day": ("time_cos", "time_sin"),
        }
        actual = {}
        for name, definition in INDICATORS.items():
            if name == "ema":
                actual[name] = tuple(definition.output_name.format(span=value)
                                      for value in sorted(definition.parameters["span"]))
            elif name == "return":
                actual[name] = tuple(definition.output_name.format(period=value)
                                      for value in sorted(definition.parameters["period"]))
            elif name == "volatility":
                actual[name] = tuple(definition.output_name.format(period=value)
                                      for value in sorted(definition.parameters["period"]))
            elif name == "time_of_day":
                actual[name] = tuple(definition.output_name.format(component=value)
                                      for value in sorted(definition.parameters["component"]))
            else:
                actual[name] = (definition.output_name,)
        self.assertEqual(actual, expected)

    def test_minimum_warmup_by_supported_indicator(self):
        cases = [
            (IndicatorInput("ema", {"span": 9}), "ema_9", 45),
            (IndicatorInput("ema", {"span": 21}), "ema_21", 105),
            (IndicatorInput("ema", {"span": 50}), "ema_50", 250),
            (IndicatorInput("ema_gap"), "ema_gap", 105),
            (IndicatorInput("ema_long_gap"), "ema_long_gap", 250),
            (IndicatorInput("return", {"period": 1}), "return_1", 2),
            (IndicatorInput("return", {"period": 60}), "return_60", 61),
            (IndicatorInput("rsi", {"period": 14}), "rsi_14", 15),
            (IndicatorInput("volatility", {"period": 15}), "volatility_15", 16),
            (IndicatorInput("volatility", {"period": 60}), "volatility_60", 61),
            (IndicatorInput("candle_range"), "candle_range", 1),
            (IndicatorInput("candle_body"), "candle_body", 1),
            (IndicatorInput("time_of_day", {"component": "sin"}), "time_sin", 1),
        ]
        for indicator, output, warmup in cases:
            with self.subTest(indicator=indicator.name, parameters=dict(indicator.parameters)):
                rule = RuleNode("gt", (FeatureRef(output), 0.0))
                hypothesis = make_hypothesis(
                    indicators=(indicator,), long_entry=rule,
                    short_entry=RuleNode("lt", (FeatureRef(output), 0.0)),
                    required_warmup_bars=warmup,
                )
                report = validate_hypothesis(hypothesis)
                self.assertEqual(report.minimum_warmup_bars, warmup)
                self.assertEqual(report.indicator_warmup_bars[output], warmup)

    def test_every_allowed_operator_is_accepted_with_valid_shape(self):
        feature = FeatureRef("rsi_14")
        compare = RuleNode("gt", (feature, 30.0))
        rule_by_operator = {
            **{op: RuleNode(op, (feature, 30.0)) for op in COMPARISON_OPERATORS},
            "crosses_above": RuleNode("crosses_above", (feature, 30.0)),
            "crosses_below": RuleNode("crosses_below", (feature, 70.0)),
            "and": RuleNode("and", (compare, RuleNode("lt", (feature, 80.0)))),
            "or": RuleNode("or", (compare, RuleNode("lt", (feature, 20.0)))),
            "not": RuleNode("not", (compare,)),
        }
        self.assertEqual(set(rule_by_operator), set(SUPPORTED_OPERATORS))
        for operator, rule in rule_by_operator.items():
            with self.subTest(operator=operator):
                validate_hypothesis(make_hypothesis(long_entry=rule))

    def test_unknown_operator_indicator_and_parameter_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported rule operator"):
            validate_hypothesis(make_hypothesis(long_entry=RuleNode("eval", (1, 2))))
        with self.assertRaisesRegex(ValueError, "unsupported indicator"):
            validate_hypothesis(make_hypothesis(indicators=(IndicatorInput("macd", {}),)))
        with self.assertRaisesRegex(ValueError, "invalid ema.span"):
            validate_hypothesis(make_hypothesis(
                indicators=(IndicatorInput("ema", {"span": 7}),),
            ))
        with self.assertRaisesRegex(ValueError, "parameters must be exactly"):
            validate_hypothesis(make_hypothesis(
                indicators=(IndicatorInput("rsi", {"period": 14, "smooth": 2}),),
            ))

    def test_malformed_rule_arities_and_shapes_are_rejected(self):
        malformed = [
            RuleNode("gt", (FeatureRef("rsi_14"),)),
            RuleNode("and", (RuleNode("gt", (FeatureRef("rsi_14"), 1)),)),
            RuleNode("not", (RuleNode("gt", (FeatureRef("rsi_14"), 1)),
                              RuleNode("lt", (FeatureRef("rsi_14"), 99)))),
            FeatureRef("rsi_14"),
        ]
        for rule in malformed:
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                validate_hypothesis(make_hypothesis(long_entry=rule))

    def test_future_offsets_unknown_fields_and_labels_are_rejected(self):
        for ref, message in [
            (FeatureRef("rsi_14", -1), "future data"),
            (FeatureRef("rsi_14", 501), "maximum"),
            (FeatureRef("volume"), "unsupported or unavailable"),
            (FeatureRef("target"), "future/label field"),
            (FeatureRef("future_outcome_end_time"), "future/label field"),
        ]:
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, message):
                validate_hypothesis(make_hypothesis(
                    long_entry=RuleNode("gt", (ref, 1.0)),
                ))

    def test_rule_references_must_be_declared_and_used_indicators_cannot_be_dead(self):
        with self.assertRaisesRegex(ValueError, "unsupported or unavailable"):
            validate_hypothesis(make_hypothesis(
                long_entry=RuleNode("gt", (FeatureRef("ema_9"), 0)),
            ))
        with self.assertRaisesRegex(ValueError, "unused"):
            validate_hypothesis(make_hypothesis(
                indicators=(IndicatorInput("rsi", {"period": 14}),
                            IndicatorInput("ema", {"span": 9})),
            ))

    def test_crossing_requires_feature_and_adds_one_prior_observation(self):
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            validate_hypothesis(make_hypothesis(
                long_entry=RuleNode("crosses_above", (1.0, 2.0)),
            ))
        report = validate_hypothesis(make_hypothesis())
        self.assertEqual(report.minimum_warmup_bars, 16)
        with self.assertRaisesRegex(ValueError, "below minimum 16"):
            validate_hypothesis(make_hypothesis(required_warmup_bars=15))

    def test_fixed_duration_exit_and_holding_duration_validation(self):
        for kind in ("stop_loss", "take_profit", "trailing_stop", "rule_driven"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "only fixed_duration"):
                validate_hypothesis(make_hypothesis(exit_rule=ExitRule(kind, "15min")))
        for duration in ("0s", "-1min", "not-a-duration"):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                validate_hypothesis(make_hypothesis(exit_rule=ExitRule("fixed_duration", duration)))
        with self.assertRaisesRegex(ValueError, "at least one bar_duration"):
            validate_hypothesis(make_hypothesis(exit_rule=ExitRule("fixed_duration", "30s")))

    def test_execution_assumptions_are_limited_to_backtester_contract(self):
        invalid = [
            ExecutionAssumptions(timestamp_convention="local_time"),
            ExecutionAssumptions(execution_delay="-1s"),
            ExecutionAssumptions(gap_tolerance="-1s"),
            ExecutionAssumptions(transaction_cost_pips=-0.1),
            ExecutionAssumptions(pip_size=0),
            ExecutionAssumptions(allow_overlapping=1),
        ]
        for execution in invalid:
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                validate_hypothesis(make_hypothesis(execution_assumptions=execution))
        with self.assertRaisesRegex(ValueError, "supported ExecutionAssumptions"):
            validate_hypothesis(make_hypothesis(execution_assumptions={"broker": "x"}))

    def test_decision_timing_family_and_metadata_are_validated(self):
        with self.assertRaisesRegex(ValueError, "after_signal_candle_close"):
            validate_hypothesis(make_hypothesis(decision_timing="before_close"))
        with self.assertRaisesRegex(ValueError, "unsupported strategy family"):
            validate_hypothesis(make_hypothesis(strategy_family="arbitrage"))
        with self.assertRaisesRegex(ValueError, "required_warmup_bars"):
            validate_hypothesis(make_hypothesis(required_warmup_bars=15.0))

    def test_duplicate_behavior_is_detected_but_validation_is_not_performance_based(self):
        first = make_hypothesis()
        second = make_hypothesis(hypothesis_id="renamed", version="2", rationale="New wording.")
        with self.assertRaisesRegex(ValueError, "duplicate hypothesis behavior"):
            validate_unique_hypotheses([first, second])
        validate_unique_hypotheses([first])


if __name__ == "__main__":
    unittest.main()
