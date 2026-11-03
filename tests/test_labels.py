"""Synthetic unit tests for causal labels and purged chronological splits."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.data import load_market_data
from src.research.features import build_features
from src.research.labels import build_labeled_samples, chronological_split


def make_market_data(size=150, *, missing_index=None):
    times = pd.date_range("2025-01-01", periods=size, freq="min", tz="UTC")
    close = 1.1 + 0.0002 * np.sin(np.arange(size) / 4) + np.arange(size) * 0.000001
    frame = pd.DataFrame({
        "time": times,
        "open": close - 0.00001,
        "high": close + 0.00002,
        "low": close - 0.00002,
        "close": close,
        "volume": np.ones(size),
    })
    if missing_index is not None:
        frame = frame.drop(index=missing_index).reset_index(drop=True)
    return frame


class LabelTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = make_market_data()
        cls.decision_index = 70  # Beyond feature warm-up.

    def sample_for_decision(self, samples, timestamp):
        selected = samples.loc[samples["decision_timestamp"] == timestamp]
        self.assertEqual(len(selected), 1)
        return selected.iloc[0]

    def test_entry_is_strictly_after_decision_and_delay_is_applied(self):
        decision_time = self.market.loc[self.decision_index, "time"] + pd.Timedelta("1min")
        immediate, _ = build_labeled_samples(
            self.market, timestamp_convention="bar_open", execution_delay="0s"
        )
        delayed, _ = build_labeled_samples(
            self.market, timestamp_convention="bar_open", execution_delay="1min"
        )
        immediate_row = self.sample_for_decision(immediate, decision_time)
        delayed_row = self.sample_for_decision(delayed, decision_time)

        self.assertGreater(immediate_row.entry_timestamp, immediate_row.decision_timestamp)
        self.assertEqual(immediate_row.entry_timestamp, self.market.loc[self.decision_index + 2, "time"])
        self.assertEqual(delayed_row.entry_timestamp, self.market.loc[self.decision_index + 3, "time"])
        self.assertEqual(immediate_row.entry_price, self.market.loc[self.decision_index + 2, "open"])

    def test_close_labeled_convention_uses_first_open_strictly_after_decision(self):
        samples, _ = build_labeled_samples(
            self.market, timestamp_convention="bar_close", execution_delay="0s"
        )
        decision_time = self.market.loc[self.decision_index, "time"]
        row = self.sample_for_decision(samples, decision_time)
        expected_entry_index = self.decision_index + 2
        expected_open_time = self.market.loc[expected_entry_index, "time"] - pd.Timedelta("1min")

        self.assertGreater(row.entry_timestamp, row.decision_timestamp)
        self.assertEqual(row.entry_timestamp, expected_open_time)
        self.assertEqual(row.entry_price, self.market.loc[expected_entry_index, "open"])

    def test_horizon_is_anchored_to_actual_entry_and_target_matches_prices(self):
        samples, _ = build_labeled_samples(
            self.market,
            horizon="15min",
            timestamp_convention="bar_open",
            execution_delay="1min",
        )
        row = self.sample_for_decision(
            samples,
            self.market.loc[self.decision_index, "time"] + pd.Timedelta("1min"),
        )

        self.assertEqual(row.exit_timestamp, row.entry_timestamp + pd.Timedelta("15min"))
        self.assertEqual(row.future_outcome_end_time, row.exit_timestamp)
        self.assertEqual(row.target, int(row.exit_price > row.entry_price))
        self.assertTrue((samples["target"] == (samples["exit_price"] > samples["entry_price"]).astype("int8")).all())
        self.assertEqual(set(samples["target"].unique()), {0, 1})

    def test_gap_spanning_samples_are_rejected(self):
        market = make_market_data(size=150, missing_index=90)
        samples, counts = build_labeled_samples(
            market,
            timestamp_convention="bar_open",
            gap_tolerance="90s",
        )
        decision_time = market.loc[75, "time"] + pd.Timedelta("1min")

        self.assertGreater(counts["rejected_by_reason"]["gap_exceeds_tolerance"], 0)
        self.assertFalse((samples["decision_timestamp"] == decision_time).any())

    def test_labeler_and_loader_reject_malformed_ohlc_prices(self):
        invalid_values = (np.nan, np.inf, 0.0, -1.0)
        for column in ("open", "high", "low", "close"):
            for invalid_price in invalid_values:
                with self.subTest(column=column, invalid_price=invalid_price):
                    market = self.market.copy()
                    market.loc[75, column] = invalid_price
                    with self.assertRaisesRegex(ValueError, "OHLC"):
                        build_labeled_samples(
                            market, timestamp_convention="bar_open"
                        )

                    with TemporaryDirectory() as temp_dir:
                        csv_path = Path(temp_dir) / "synthetic.csv"
                        market.to_csv(csv_path, index=False)
                        with self.assertRaisesRegex(ValueError, "OHLC"):
                            load_market_data(csv_path)

    def test_loader_reports_dropped_and_duplicate_rows(self):
        market = self.market.iloc[:5].copy().reset_index(drop=True)
        market.loc[0, "time"] = pd.NaT
        market.loc[1, "volume"] = np.nan
        market.loc[4, "time"] = market.loc[2, "time"]
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "synthetic.csv"
            market.to_csv(csv_path, index=False)
            loaded = load_market_data(csv_path)

        report = loaded.attrs["load_report"]
        self.assertEqual(report["input_rows"], 5)
        self.assertEqual(report["dropped_missing_timestamps"], 1)
        self.assertEqual(report["dropped_missing_volume"], 1)
        self.assertEqual(report["dropped_duplicate_timestamps"], 1)
        self.assertEqual(report["output_rows"], 2)

    def test_loader_localizes_naive_csv_timestamps_as_utc_without_shifting(self):
        market = self.market.iloc[:4].copy()
        expected = market["time"].copy()
        market["time"] = market["time"].dt.tz_localize(None)
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "naive_utc.csv"
            market.to_csv(csv_path, index=False)
            loaded = load_market_data(csv_path, naive_timezone="UTC")

        self.assertEqual(str(loaded["time"].dt.tz), "UTC")
        pd.testing.assert_series_equal(loaded["time"], expected, check_names=False)

        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "unknown_naive_zone.csv"
            market.to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "Naive timestamps require"):
                load_market_data(csv_path, naive_timezone=None)

    def test_loader_converts_aware_inputs_to_utc_and_rejects_mixed_awareness(self):
        market = self.market.iloc[:4].copy()
        market["time"] = market["time"].dt.tz_convert("Europe/Budapest")
        expected = self.market["time"].iloc[:4].reset_index(drop=True)
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "aware.csv"
            market.to_csv(csv_path, index=False)
            loaded = load_market_data(csv_path)
            pd.testing.assert_series_equal(
                loaded["time"], expected, check_names=False
            )

            mixed = market.copy()
            mixed["time"] = [
                self.market["time"].iloc[0].tz_localize(None),
                market["time"].iloc[1],
                market["time"].iloc[2],
                market["time"].iloc[3],
            ]
            mixed_path = Path(temp_dir) / "mixed.csv"
            mixed.to_csv(mixed_path, index=False)
            with self.assertRaisesRegex(ValueError, "Mixed timezone-aware and naive"):
                load_market_data(mixed_path)

    def test_features_and_labels_keep_timezone_aware_utc_timestamps(self):
        local_market = self.market.copy()
        local_market["time"] = local_market["time"].dt.tz_convert("Europe/Budapest")
        featured = build_features(local_market)
        samples, _ = build_labeled_samples(
            local_market, timestamp_convention="bar_open"
        )

        self.assertEqual(str(featured["time"].dt.tz), "UTC")
        self.assertEqual(str(samples["decision_timestamp"].dt.tz), "UTC")
        row = self.sample_for_decision(
            samples,
            self.market.loc[self.decision_index, "time"] + pd.Timedelta("1min"),
        )
        self.assertEqual(
            row.entry_timestamp,
            self.market.loc[self.decision_index + 2, "time"],
        )

    def test_features_reject_naive_timestamps_without_source_timezone_context(self):
        market = self.market.copy()
        market["time"] = market["time"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "Naive timestamps require"):
            build_features(market)


class SplitPurgingTests(unittest.TestCase):
    def test_train_and_validation_outcomes_do_not_cross_next_boundary(self):
        decision_times = pd.date_range("2025-01-01", periods=50, freq="min", tz="UTC")
        samples = pd.DataFrame({
            "decision_timestamp": decision_times,
            "future_outcome_end_time": decision_times + pd.Timedelta("5min"),
        })
        splits = chronological_split(
            samples,
            train_fraction=0.60,
            validation_fraction=0.20,
            purge_interval="0min",
        )
        validation_start = splits["validation"]["decision_timestamp"].min()
        test_start = splits["test"]["decision_timestamp"].min()

        self.assertFalse(splits["train"].empty)
        self.assertFalse(splits["validation"].empty)
        self.assertTrue((splits["train"]["future_outcome_end_time"] < validation_start).all())
        self.assertTrue((splits["validation"]["future_outcome_end_time"] < test_start).all())


if __name__ == "__main__":
    unittest.main()
