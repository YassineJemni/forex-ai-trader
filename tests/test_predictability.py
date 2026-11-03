import unittest

import numpy as np
import pandas as pd

from src.research.labels import MID_RETURN_FEATURE_COLUMNS, MID_RETURN_HORIZONS
from src.research.predictability import (
    FAMILYWISE_ALPHA,
    MAP_FAMILY_ID,
    SHIFT_RESAMPLES,
    evaluate_predictability_map,
)


BOUNDARY = "2025-01-03T00:00:00Z"


def samples_for(rows=1200, *, seed=1, planted=False):
    rng = np.random.default_rng(seed)
    decisions = pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC")
    frame = pd.DataFrame({
        "decision_timestamp": np.tile(decisions, len(MID_RETURN_HORIZONS)),
        "horizon": np.repeat(MID_RETURN_HORIZONS, rows),
        "accepted": True,
    })
    frame["entry_timestamp"] = frame["decision_timestamp"] + pd.Timedelta("1min")
    horizon_delta = frame["horizon"].map(pd.Timedelta)
    frame["exit_timestamp"] = frame["entry_timestamp"] + horizon_delta

    feature_values = {
        feature: rng.normal(size=rows) for feature in MID_RETURN_FEATURE_COLUMNS
    }
    for feature, values in feature_values.items():
        frame[feature] = np.tile(values, len(MID_RETURN_HORIZONS))
    frame["forward_log_return"] = rng.normal(size=len(frame))
    if planted:
        target = frame["horizon"].eq("5min")
        x = frame.loc[target, "return_1"].to_numpy()
        frame.loc[target, "forward_log_return"] = x + rng.normal(scale=0.15, size=rows)
    return frame


class PredictabilityMapTests(unittest.TestCase):
    def test_family_is_fixed_complete_and_reproducible(self):
        samples = samples_for(rows=400)
        first = evaluate_predictability_map(samples, training_boundary_utc=BOUNDARY)
        second = evaluate_predictability_map(samples, training_boundary_utc=BOUNDARY)
        self.assertEqual(first["family_id"], MAP_FAMILY_ID)
        self.assertEqual(first["family"]["size"], 64)
        self.assertEqual(first["family"]["feature_horizon_tests"], 64)
        self.assertEqual(len(first["results"]), 64)
        self.assertEqual(first, second)
        self.assertEqual({row["n"] for row in first["results"]}, {400})
        self.assertEqual(first["inference"]["shift_resamples"], SHIFT_RESAMPLES)
        self.assertEqual(first["inference"]["multiple_testing"].split()[0], "Holm")
        self.assertEqual(
            {row["horizon"] for row in first["results"]},
            set(MID_RETURN_HORIZONS),
        )

    def test_planted_predictive_signal_is_detected(self):
        result = evaluate_predictability_map(
            samples_for(rows=1000, seed=7, planted=True),
            training_boundary_utc=BOUNDARY,
        )
        planted = next(
            row for row in result["results"]
            if row["feature"] == "return_1" and row["horizon"] == "5min"
        )
        self.assertGreater(planted["correlation"], 0.9)
        self.assertLessEqual(planted["holm_adjusted_p_value"], FAMILYWISE_ALPHA)
        self.assertGreaterEqual(planted["hac_95_ci"][0], 0.0)

    def test_shift_null_control_controls_familywise_false_positives(self):
        false_positive_families = 0
        simulation_count = 30
        for seed in range(simulation_count):
            result = evaluate_predictability_map(
                samples_for(rows=300, seed=seed + 100),
                training_boundary_utc=BOUNDARY,
            )
            if any(
                row["shift_max_statistic_fwer_p_value"] <= FAMILYWISE_ALPHA
                for row in result["results"]
            ):
                false_positive_families += 1
        self.assertLessEqual(false_positive_families, 5)

    def test_historical_rows_at_or_after_boundary_are_rejected(self):
        samples = samples_for(rows=400)
        samples.loc[0, "exit_timestamp"] = pd.Timestamp(BOUNDARY)
        with self.assertRaisesRegex(ValueError, "strictly before the frozen training boundary"):
            evaluate_predictability_map(samples, training_boundary_utc=BOUNDARY)


if __name__ == "__main__":
    unittest.main()
