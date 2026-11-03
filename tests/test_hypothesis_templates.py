"""Deterministic bounded template-generation tests; no market data is used."""

import ast
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from src.research.hypothesis_schema import RuleNode
import src.research.hypothesis_templates as templates
from src.research.hypothesis_templates import (
    DEFAULT_CANDIDATE_LIMIT,
    MAX_CANDIDATE_LIMIT,
    GenerationResult,
    TemplateConfig,
    generate_hypotheses,
)
from src.research.hypothesis_validator import validate_hypothesis


class HypothesisTemplateTests(unittest.TestCase):
    def test_identical_configuration_produces_identical_order_and_content(self):
        config = TemplateConfig(candidate_limit=9)
        one = generate_hypotheses(config)
        two = generate_hypotheses(config)
        self.assertEqual(one, two)
        self.assertEqual(
            [item.hypothesis_id for item in one.hypotheses],
            [item.hypothesis_id for item in two.hypotheses],
        )

    def test_default_grammar_emits_each_predeclared_family(self):
        result = generate_hypotheses()
        self.assertEqual(
            {item.strategy_family for item in result.hypotheses},
            {"trend_following", "mean_reversion", "breakout", "momentum", "volatility_filter"},
        )
        self.assertEqual(result.proposals_generated, result.candidate_space_size)
        self.assertTrue(all(validate_hypothesis(item).semantic_digest == item.semantic_digest()
                            for item in result.hypotheses))
        breakout = next(item for item in result.hypotheses if item.strategy_family == "breakout")
        self.assertIn("previous candle", breakout.rationale)
        self.assertNotIn("donchian", breakout.rationale.lower())

    def test_candidate_limit_is_configurable_and_reports_unvisited_candidates(self):
        config = TemplateConfig(candidate_limit=3)
        result = generate_hypotheses(config)
        self.assertEqual(result.proposals_generated, 3)
        self.assertEqual(len(result.hypotheses), 3)
        self.assertEqual(result.not_generated_due_to_limit, result.candidate_space_size - 3)
        self.assertEqual(DEFAULT_CANDIDATE_LIMIT, 50)
        self.assertEqual(MAX_CANDIDATE_LIMIT, 100)
        with self.assertRaisesRegex(ValueError, "candidate_limit"):
            TemplateConfig(candidate_limit=MAX_CANDIDATE_LIMIT + 1)
        with self.assertRaisesRegex(ValueError, "candidate_limit"):
            TemplateConfig(candidate_limit=0)

    def test_family_selection_and_enumeration_order_are_stable(self):
        config = TemplateConfig(families=("momentum", "trend_following"), candidate_limit=20)
        result = generate_hypotheses(config)
        families = [item.strategy_family for item in result.hypotheses]
        self.assertEqual(families, ["trend_following"] * 3 + ["momentum"] * 3)

    def test_validator_is_called_for_each_proposal(self):
        original = templates.validate_hypothesis
        with patch.object(templates, "validate_hypothesis", wraps=original) as validate:
            result = generate_hypotheses(TemplateConfig(candidate_limit=4))
        self.assertEqual(validate.call_count, result.proposals_generated)

    def test_invalid_proposal_is_counted_and_reason_is_preserved(self):
        valid = next(iter(templates._iter_proposals(TemplateConfig(
            families=("mean_reversion",), rsi_threshold_pairs=((30.0, 70.0),),
        ))))
        invalid = replace(valid, long_entry=RuleNode("exec", (1, 2)))
        config = TemplateConfig(
            families=("mean_reversion",), rsi_threshold_pairs=((30.0, 70.0),),
        )
        changed_display = replace(
            valid, hypothesis_id="different_display_id", rationale="Different proposal wording.",
        )
        with patch.object(templates, "_iter_proposals", return_value=iter((invalid,))):
            result = generate_hypotheses(config)
        self.assertEqual(result.proposals_generated, 1)
        self.assertEqual(result.proposals_validated, 0)
        self.assertEqual(result.proposals_rejected, 1)
        self.assertEqual(result.proposals_deduplicated, 0)
        self.assertEqual(result.hypotheses, ())
        self.assertEqual(len(result.rejections), 1)
        self.assertIn("unsupported rule operator", result.rejections[0].reason)

        with patch.object(templates, "_iter_proposals", return_value=iter((valid, changed_display))):
            duplicate_result = generate_hypotheses(
                TemplateConfig(families=("mean_reversion",), rsi_threshold_pairs=(
                    (30.0, 70.0), (30.0, 70.0),
                )),
            )
        self.assertEqual(duplicate_result.proposals_deduplicated, 1)
        self.assertEqual(len(duplicate_result.hypotheses), 1)

    def test_semantically_duplicate_proposals_are_counted_and_first_is_kept(self):
        config = TemplateConfig(
            families=("mean_reversion",),
            rsi_threshold_pairs=((30.0, 70.0), (30.0, 70.0)),
        )
        result = generate_hypotheses(config)
        self.assertEqual(result.proposals_generated, 2)
        self.assertEqual(result.proposals_validated, 2)
        self.assertEqual(result.proposals_rejected, 0)
        self.assertEqual(result.proposals_deduplicated, 1)
        self.assertEqual(len(result.hypotheses), 1)

    def test_template_parameters_are_bounded_and_not_arbitrary(self):
        with self.assertRaisesRegex(ValueError, "ema_pairs"):
            TemplateConfig(ema_pairs=((9, 22),))
        with self.assertRaisesRegex(ValueError, "rsi_threshold_pairs"):
            TemplateConfig(rsi_threshold_pairs=((10.0, 90.0),))
        with self.assertRaisesRegex(ValueError, "momentum_periods"):
            TemplateConfig(momentum_periods=(7,))
        with self.assertRaisesRegex(ValueError, "volatility_thresholds"):
            TemplateConfig(volatility_thresholds=(0.01,))
        with self.assertRaisesRegex(ValueError, "unsupported template families"):
            TemplateConfig(families=("arbitrage",))

    def test_causal_trend_volatility_composite_is_optional_and_validated(self):
        config = TemplateConfig(
            families=("trend_following",), ema_pairs=((9, 21),),
            include_trend_volatility_filters=True, candidate_limit=2,
        )
        result = generate_hypotheses(config)
        composite = [hypothesis for hypothesis in result.hypotheses
                     if len(hypothesis.indicators) == 3]
        self.assertEqual(len(composite), 1)
        self.assertEqual(composite[0].required_warmup_bars, 106)
        self.assertEqual(validate_hypothesis(composite[0]).semantic_digest,
                         composite[0].semantic_digest())
        self.assertFalse(TemplateConfig().include_trend_volatility_filters)

    def test_module_has_no_data_experiment_or_execution_import_dependency(self):
        source = Path(templates.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")
        forbidden = {
            "src.research.data", "src.research.experiment_store",
            "src.research.experiment_orchestrator", "src.research.compare_strategies",
            "src.research.backtester",
        }
        self.assertFalse(imported_modules & forbidden)
        self.assertNotIn("results", source)
        self.assertNotIn("test_period", source)


if __name__ == "__main__":
    unittest.main()
