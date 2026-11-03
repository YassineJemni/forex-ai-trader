import unittest

import numpy as np
import pandas as pd

from src.research.execution_models import MarketDataMetadata
from src.research.labels import (
    TrainingBoundarySpec,
    build_mid_return_samples,
)


def market_data(rows=500, *, missing_indices=()):
    times = pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC")
    mid = 1.1 + np.arange(rows, dtype=float) * 0.00001
    bid = pd.DataFrame({
        "time": times,
        "open": mid,
        "high": mid + 0.00004,
        "low": mid - 0.00004,
        "close": mid + 0.00001,
        "volume": np.ones(rows),
    }).drop(index=list(missing_indices)).reset_index(drop=True)
    ask = bid.copy()
    ask[["open", "high", "low", "close"]] += 0.00002
    return bid, ask


def boundary_spec(bid, *, training_rows=450, timestamp=None):
    observed = bid["time"].sort_values().reset_index(drop=True).iloc[training_rows]
    stamp = observed if timestamp is None else pd.Timestamp(timestamp)
    return TrainingBoundarySpec(
        training_rows=training_rows,
        boundary_timestamp_utc=stamp.isoformat().replace("+00:00", "Z"),
    )


def run_labels(bid, ask, *, spec=None, horizons=("1min", "5min", "15min", "60min")):
    return build_mid_return_samples(
        bid,
        ask,
        bid_metadata=MarketDataMetadata("BID", "EUR/USD", "M1", "bar_open", "UTC", "synthetic"),
        ask_metadata=MarketDataMetadata("ASK", "EUR/USD", "M1", "bar_open", "UTC", "synthetic"),
        boundary_spec=spec or boundary_spec(bid),
        horizons=horizons,
    )


class MidReturnLabelTests(unittest.TestCase):
    def test_exact_entry_exit_and_log_return_for_one_and_five_minutes(self):
        bid, ask = market_data()
        samples, audit = run_labels(bid, ask, horizons=("1min", "5min"))
        decision = pd.Timestamp("2025-01-01T00:11:00Z")
        rows = samples.loc[samples["decision_timestamp"] == decision].set_index("horizon")

        self.assertEqual(rows.loc["1min", "entry_timestamp"], pd.Timestamp("2025-01-01T00:12:00Z"))
        self.assertEqual(
            rows.loc["1min", "exit_timestamp"],
            pd.Timestamp("2025-01-01T00:13:00Z"),
        )
        self.assertEqual(
            rows.loc["5min", "exit_timestamp"],
            pd.Timestamp("2025-01-01T00:17:00Z"),
        )
        entry_mid = (float(bid.loc[12, "open"]) + float(ask.loc[12, "open"])) / 2
        one_exit_mid = (float(bid.loc[12, "close"]) + float(ask.loc[12, "close"])) / 2
        five_exit_mid = (float(bid.loc[16, "close"]) + float(ask.loc[16, "close"])) / 2
        self.assertAlmostEqual(rows.loc["1min", "forward_log_return"], np.log(one_exit_mid / entry_mid))
        self.assertAlmostEqual(rows.loc["5min", "forward_log_return"], np.log(five_exit_mid / entry_mid))
        self.assertTrue(rows["accepted"].all())
        self.assertEqual(audit["boundary"]["verified_index"], 450)
        self.assertFalse(audit["boundary"]["post_boundary_prices_used"])

    def test_mismatched_entry_quote_is_explicitly_rejected(self):
        bid, ask = market_data()
        ask = ask.loc[ask["time"] != bid.loc[12, "time"]].reset_index(drop=True)
        samples, audit = run_labels(bid, ask, horizons=("1min",))
        decision = pd.Timestamp("2025-01-01T00:11:00Z")
        row = samples.loc[samples["decision_timestamp"] == decision].iloc[0]
        self.assertFalse(row["accepted"])
        self.assertEqual(row["rejection_reason"], "unmatched_entry_quote")
        self.assertGreater(audit["rejected_by_reason"]["unmatched_entry_quote"], 0)

    def test_mismatched_exact_exit_quote_is_explicitly_rejected(self):
        bid, ask = market_data()
        ask = ask.loc[ask["time"] != bid.loc[16, "time"]].reset_index(drop=True)
        samples, audit = run_labels(bid, ask, horizons=("5min",))
        decision = pd.Timestamp("2025-01-01T00:11:00Z")
        row = samples.loc[samples["decision_timestamp"] == decision].iloc[0]
        self.assertFalse(row["accepted"])
        self.assertEqual(row["rejection_reason"], "unmatched_exact_exit_quote")
        self.assertGreater(audit["rejected_by_reason"]["unmatched_exact_exit_quote"], 0)

    def test_missing_exact_exit_candle_is_not_replaced_by_a_later_close(self):
        bid, ask = market_data(missing_indices=(16,))
        spec = boundary_spec(bid, training_rows=449)
        samples, audit = run_labels(bid, ask, spec=spec, horizons=("5min",))
        decision = pd.Timestamp("2025-01-01T00:11:00Z")
        row = samples.loc[samples["decision_timestamp"] == decision].iloc[0]
        self.assertFalse(row["accepted"])
        self.assertEqual(row["rejection_reason"], "exact_exit_timestamp_missing")
        self.assertGreater(audit["rejected_by_reason"]["exact_exit_timestamp_missing"], 0)

    def test_gap_inside_holding_window_is_rejected(self):
        bid, ask = market_data(missing_indices=(13,))
        spec = boundary_spec(bid, training_rows=449)
        samples, _ = run_labels(bid, ask, spec=spec, horizons=("5min",))
        decision = pd.Timestamp("2025-01-01T00:11:00Z")
        row = samples.loc[samples["decision_timestamp"] == decision].iloc[0]
        self.assertFalse(row["accepted"])
        self.assertEqual(row["rejection_reason"], "gap_exceeds_tolerance")

    def test_boundary_timestamp_and_all_required_bars_are_strictly_training(self):
        bid, ask = market_data()
        spec = boundary_spec(bid)
        samples, _ = run_labels(bid, ask, spec=spec, horizons=("1min", "5min"))
        boundary = pd.Timestamp(spec.boundary_timestamp_utc)
        accepted = samples.loc[samples["accepted"]]
        self.assertTrue((accepted["decision_timestamp"] < boundary).all())
        self.assertTrue((accepted["entry_timestamp"] < boundary).all())
        self.assertTrue((accepted["exit_timestamp"] < boundary).all())
        decision = pd.Timestamp("2025-01-01T07:28:00Z")
        row = samples.loc[
            (samples["decision_timestamp"] == decision) & (samples["horizon"] == "1min")
        ].iloc[0]
        self.assertFalse(row["accepted"])
        self.assertEqual(row["rejection_reason"], "exit_at_or_after_training_boundary")

    def test_boundary_mismatch_fails_closed_before_labeling(self):
        bid, ask = market_data()
        wrong_spec = boundary_spec(bid, timestamp="2025-01-01T07:31:00Z")
        with self.assertRaisesRegex(ValueError, "cannot verify frozen training boundary"):
            run_labels(bid, ask, spec=wrong_spec)
        with self.assertRaisesRegex(ValueError, "original timeline ends before index 404889"):
            build_mid_return_samples(
                bid,
                ask,
                bid_metadata=MarketDataMetadata("BID", "EUR/USD", "M1", "bar_open", "UTC", "synthetic"),
                ask_metadata=MarketDataMetadata("ASK", "EUR/USD", "M1", "bar_open", "UTC", "synthetic"),
            )

    def test_future_truncation_does_not_change_earlier_features_or_labels(self):
        bid, ask = market_data()
        spec = boundary_spec(bid)
        baseline, _ = run_labels(bid, ask, spec=spec, horizons=("1min", "5min"))
        cutoff = bid.loc[250, "time"]
        changed_bid, changed_ask = bid.copy(), ask.copy()
        changed_bid.loc[changed_bid["time"] > cutoff, ["open", "high", "low", "close"]] += 0.25
        changed_ask.loc[changed_ask["time"] > cutoff, ["open", "high", "low", "close"]] += 0.25
        changed, _ = run_labels(changed_bid, changed_ask, spec=spec, horizons=("1min", "5min"))
        stable_before_cutoff = (
            (baseline["decision_timestamp"] <= cutoff)
            & (baseline["exit_timestamp"] < cutoff)
        )
        pd.testing.assert_frame_equal(
            baseline.loc[stable_before_cutoff].reset_index(drop=True),
            changed.loc[stable_before_cutoff].reset_index(drop=True),
        )

    def test_post_boundary_prices_are_not_validated_or_used(self):
        bid, ask = market_data()
        spec = boundary_spec(bid)
        baseline, _ = run_labels(bid, ask, spec=spec, horizons=("1min",))
        bid.loc[475, ["open", "high", "low", "close"]] = -1.0
        ask.loc[475, ["open", "high", "low", "close"]] = -1.0
        after, _ = run_labels(bid, ask, spec=spec, horizons=("1min",))
        pd.testing.assert_frame_equal(baseline, after)


if __name__ == "__main__":
    unittest.main()
