"""Serialization, immutability, and canonical identity tests for hypotheses."""

import unittest

from src.research.hypothesis_schema import (
    ExecutionAssumptions,
    ExitRule,
    FeatureRef,
    IndicatorInput,
    RuleNode,
    StrategyHypothesis,
)


def make_hypothesis(**changes):
    values = {
        "hypothesis_id": "rsi_recross_a",
        "version": "1.0.0",
        "strategy_family": "mean_reversion",
        "rationale": "Descriptive rationale.",
        "indicators": (IndicatorInput("rsi", {"period": 14}),),
        "long_entry": RuleNode("crosses_above", (FeatureRef("rsi_14"), 30.0)),
        "short_entry": RuleNode("crosses_below", (FeatureRef("rsi_14"), 70.0)),
        "decision_timing": "after_signal_candle_close",
        "exit_rule": ExitRule("fixed_duration", "15min"),
        "required_warmup_bars": 16,
        "execution_assumptions": ExecutionAssumptions(),
        "assumptions": ("Fixed cost is an assumption.",),
        "known_limitations": ("BID-only price inputs.",),
    }
    values.update(changes)
    return StrategyHypothesis(**values)


class HypothesisSchemaTests(unittest.TestCase):
    def test_to_dict_from_dict_round_trip(self):
        original = make_hypothesis()
        restored = StrategyHypothesis.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_parameter_mappings_are_recursively_immutable(self):
        source = {"period": 14, "nested": {"allowed": [1, 2]}}
        indicator = IndicatorInput("rsi", source)
        source["period"] = 99
        self.assertEqual(indicator.parameters["period"], 14)
        self.assertEqual(indicator.parameters["nested"]["allowed"], (1, 2))
        with self.assertRaises(TypeError):
            indicator.parameters["period"] = 20

    def test_canonical_json_is_sorted_and_compact(self):
        one = make_hypothesis()
        reordered_params = IndicatorInput("rsi", {"period": 14})
        two = make_hypothesis(indicators=(reordered_params,))
        self.assertEqual(one.canonical_json(), two.canonical_json())
        self.assertNotIn("hypothesis_id", one.canonical_json())
        self.assertNotIn("rationale", one.canonical_json())
        self.assertEqual(len(one.semantic_digest()), 64)

    def test_semantic_digest_excludes_ids_versions_and_display_text(self):
        first = make_hypothesis()
        renamed = make_hypothesis(
            hypothesis_id="another_id", version="2.0", rationale="Different wording.",
            assumptions=("Different explanatory assumption text.",),
            known_limitations=("Different display note.",),
        )
        self.assertEqual(first.semantic_digest(), renamed.semantic_digest())

    def test_semantic_digest_changes_for_behavior_fields(self):
        base = make_hypothesis()
        variants = [
            make_hypothesis(long_entry=RuleNode("crosses_above", (FeatureRef("rsi_14"), 25.0))),
            make_hypothesis(exit_rule=ExitRule("fixed_duration", "20min")),
            make_hypothesis(required_warmup_bars=17),
            make_hypothesis(execution_assumptions=ExecutionAssumptions(transaction_cost_pips=1.1)),
        ]
        for variant in variants:
            with self.subTest(behavior=variant.canonical_behavior()):
                self.assertNotEqual(base.semantic_digest(), variant.semantic_digest())

    def test_family_is_classification_not_part_of_semantic_execution_digest(self):
        self.assertEqual(
            make_hypothesis().semantic_digest(),
            make_hypothesis(strategy_family="momentum").semantic_digest(),
        )

    def test_boolean_rule_canonicalization_sorts_commutative_children(self):
        a = RuleNode("gt", (FeatureRef("rsi_14"), 50))
        b = RuleNode("lt", (FeatureRef("rsi_14"), 80))
        left = make_hypothesis(long_entry=RuleNode("and", (a, b)))
        right = make_hypothesis(long_entry=RuleNode("and", (b, a)))
        self.assertEqual(left.semantic_digest(), right.semantic_digest())

    def test_numerically_equivalent_scalars_have_same_digest(self):
        integer_rule = make_hypothesis(
            long_entry=RuleNode("crosses_above", (FeatureRef("rsi_14"), 30)),
            execution_assumptions=ExecutionAssumptions(transaction_cost_pips=1),
        )
        float_rule = make_hypothesis(
            long_entry=RuleNode("crosses_above", (FeatureRef("rsi_14"), 30.0)),
            execution_assumptions=ExecutionAssumptions(transaction_cost_pips=1.0),
        )
        self.assertEqual(integer_rule.semantic_digest(), float_rule.semantic_digest())

    def test_rule_operands_cannot_hold_mutable_arbitrary_objects(self):
        with self.assertRaisesRegex(TypeError, "rule operands"):
            RuleNode("eq", ({"python": "object"}, 1))

    def test_deserialization_rejects_unknown_schema_fields(self):
        payload = make_hypothesis().to_dict()
        payload["executable_code"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "unknown=.*executable_code"):
            StrategyHypothesis.from_dict(payload)

    def test_semantic_digest_normalizes_equivalent_durations(self):
        first = make_hypothesis()
        equivalent = make_hypothesis(
            exit_rule=ExitRule("fixed_duration", "0 days 00:15:00"),
            execution_assumptions=ExecutionAssumptions(
                bar_duration="60s", execution_delay="0 days 00:00:00",
                gap_tolerance="0 days 00:01:30",
            ),
        )
        self.assertEqual(first.semantic_digest(), equivalent.semantic_digest())


if __name__ == "__main__":
    unittest.main()
