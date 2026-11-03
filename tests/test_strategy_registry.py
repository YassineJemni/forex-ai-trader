"""Tests for stable strategy metadata registration."""

import unittest

from dataclasses import replace

from src.research.strategy_registry import (
    STRATEGIES,
    StrategySpec,
    get_strategy,
    list_strategies,
    validate_registry_configuration,
)


class StrategyRegistryTests(unittest.TestCase):
    def test_existing_strategy_ids_are_unique_and_stable(self):
        specs = list_strategies()
        ids = [spec.strategy_id for spec in specs]
        self.assertEqual(ids, ["ema_crossover", "rsi_mean_reversion", "donchian_breakout"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(get_strategy("ema_crossover"), specs[0])

    def test_every_spec_has_required_metadata_and_finite_json_parameters(self):
        for spec in list_strategies():
            with self.subTest(strategy_id=spec.strategy_id):
                self.assertTrue(spec.name)
                self.assertTrue(spec.family)
                self.assertTrue(spec.hypothesis)
                self.assertTrue(spec.rule_description)
                self.assertTrue(spec.version)
                self.assertTrue(spec.parameters)

    def test_invalid_ids_and_parameter_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strategy_id"):
            StrategySpec("EMA-9-21", "name", "family", "why", "rule", {}, "1")
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            StrategySpec("valid_id", "name", "family", "why", "rule", {"x": float("nan")}, "1")
        with self.assertRaisesRegex(KeyError, "Unregistered"):
            get_strategy("unknown")

    def test_runner_configuration_must_match_registered_parameters(self):
        current = {spec.strategy_id: dict(spec.parameters) for spec in STRATEGIES}
        validate_registry_configuration(current)
        changed = dict(current)
        changed["ema_crossover"] = {**current["ema_crossover"], "fast_ema": 10}
        altered_specs = tuple(
            replace(spec, parameters={**spec.parameters, "fast_ema": 10})
            if spec.strategy_id == "ema_crossover" else spec
            for spec in STRATEGIES
        )
        with self.assertRaisesRegex(ValueError, "configuration drift.*fast_ema"):
            validate_registry_configuration(changed, specs=STRATEGIES)
        # Runner drift against a deliberately altered registry is caught too.
        with self.assertRaisesRegex(ValueError, "configuration drift.*fast_ema"):
            validate_registry_configuration(current, specs=altered_specs)


if __name__ == "__main__":
    unittest.main()
