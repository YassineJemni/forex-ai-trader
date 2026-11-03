"""Synthetic tests for the predeclared strategy comparison."""

import unittest

import numpy as np
import pandas as pd

from src.research.compare_strategies import (
    STRATEGIES,
    generate_strategy_signals,
    render_report,
)
from src.research.experiment_ema import (
    evaluate_period,
    timeline_periods,
    validate_signal_attribution,
)
from src.research.features import (
    build_features,
    indicator_segment_ids,
    segment_bar_counts,
)


def market_from_closes(closes, *, high_pad=0.00005, low_pad=0.00005):
    closes = np.asarray(closes, dtype=float)
    times = pd.date_range("2025-01-01", periods=len(closes), freq="min", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": closes,
        "high": closes + high_pad,
        "low": closes - low_pad,
        "close": closes,
        "volume": np.ones(len(closes)),
    })


def flat_market(size=240):
    return market_from_closes(np.full(size, 1.1))


class StrategySignalTests(unittest.TestCase):
    def test_all_signals_are_decided_after_the_source_candle_closes(self):
        closes = [1.2]
        for step in [-0.0001] * 30 + [0.0001] * 30 + [-0.0001] * 30:
            closes.append(closes[-1] + step)
        market = market_from_closes(closes)
        signals = generate_strategy_signals(market)

        for strategy in STRATEGIES:
            for signal in signals[strategy].itertuples(index=False):
                source = market.iloc[signal.source_index]
                self.assertEqual(signal.decision_timestamp, source.time + pd.Timedelta("1min"))
                self.assertGreater(signal.decision_timestamp, source.time)

    def test_future_data_does_not_change_earlier_signals(self):
        closes = [1.2]
        for step in [-0.0001] * 30 + [0.0001] * 30 + [-0.0001] * 30:
            closes.append(closes[-1] + step)
        market = market_from_closes(closes)
        baseline = generate_strategy_signals(market)
        repeated = generate_strategy_signals(market)
        cutoff = market.loc[45, "time"]
        changed = market.copy()
        changed.loc[46:, ["open", "high", "low", "close"]] += 0.02
        later = generate_strategy_signals(changed)

        for strategy in STRATEGIES:
            self.assertEqual(
                baseline[strategy]["signal_id"].tolist(),
                repeated[strategy]["signal_id"].tolist(),
            )
            left = baseline[strategy].loc[
                baseline[strategy]["decision_timestamp"] <= cutoff
            ].reset_index(drop=True)
            right = later[strategy].loc[
                later[strategy]["decision_timestamp"] <= cutoff
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right)

    def test_future_data_does_not_change_earlier_features(self):
        closes = [1.2]
        for step in [-0.0001] * 35 + [0.0001] * 45:
            closes.append(closes[-1] + step)
        market = market_from_closes(closes)
        baseline = build_features(market)
        changed = market.copy()
        changed.loc[65:, ["open", "high", "low", "close"]] += 0.03
        later = build_features(changed)
        feature_columns = [column for column in baseline if column not in market.columns]
        pd.testing.assert_frame_equal(
            baseline.loc[:64, feature_columns],
            later.loc[:64, feature_columns],
        )

    def test_exact_90_second_gap_stays_in_segment_but_91_seconds_resets(self):
        start = pd.Timestamp("2025-01-01", tz="UTC")
        times = pd.Series([
            start,
            start + pd.Timedelta(seconds=90),
            start + pd.Timedelta(seconds=180),
            start + pd.Timedelta(seconds=271),
        ])
        segments = indicator_segment_ids(times)
        self.assertEqual(segments.tolist(), [0, 0, 0, 1])

        market = market_from_closes([1.1, 1.1001, 1.1002, 1.1003])
        market["time"] = times
        features = build_features(market)
        self.assertFalse(pd.isna(features.loc[2, "return_1"]))
        self.assertTrue(pd.isna(features.loc[3, "return_1"]))
        self.assertEqual(features.loc[3, "ema_9"], features.loc[3, "close"])
        self.assertTrue(pd.isna(features.loc[3, "rsi_14"]))

    def test_all_indicators_reset_and_candidate_signals_rewarm_after_gap(self):
        start = pd.Timestamp("2025-01-01", tz="UTC")
        pre_count = 30
        post_count = 120
        pre_times = pd.date_range(start, periods=pre_count, freq="min")
        post_times = pd.date_range(
            pre_times[-1] + pd.Timedelta(seconds=91),
            periods=post_count,
            freq="min",
        )
        times = pre_times.append(post_times)
        closes = np.full(len(times), 1.1)
        # Two Donchian breaks in the post-gap segment: one before warm-up and
        # another once the 105-bar common warm-up has completed.
        closes[pre_count + 20] = 1.1002
        closes[pre_count + 105] = 1.1002
        market = market_from_closes(closes)
        market["time"] = times
        market.loc[:pre_count - 1, "high"] = 1.5
        market.loc[pre_count:, "high"] = np.maximum(
            market.loc[pre_count:, "high"], market.loc[pre_count:, "close"]
        )

        features = build_features(market)
        first_post = pre_count
        self.assertTrue(pd.isna(features.loc[first_post, "return_1"]))
        self.assertEqual(features.loc[first_post, "ema_9"], features.loc[first_post, "close"])
        self.assertTrue(pd.isna(features.loc[first_post, "rsi_14"]))
        self.assertTrue(pd.isna(features.loc[first_post, "volatility_15"]))
        self.assertTrue(pd.isna(features.loc[first_post, "volatility_60"]))

        donchian = generate_strategy_signals(market)[STRATEGIES[2]]
        by_source = donchian.set_index("source_row_index")
        early_break = pre_count + 20
        warmed_break = pre_count + 105
        self.assertIn(early_break, by_source.index)
        self.assertFalse(by_source.loc[early_break, "indicator_warmup_complete"])
        self.assertIn(warmed_break, by_source.index)
        self.assertTrue(by_source.loc[warmed_break, "indicator_warmup_complete"])
        self.assertEqual(by_source.loc[early_break, "indicator_segment_id"], 1)
        self.assertEqual(by_source.loc[warmed_break, "segment_bar_count"], 106)

    def test_rsi_signals_are_only_threshold_recrosses(self):
        closes = [1.2]
        for step in [-0.0001] * 30 + [0.0001] * 30 + [-0.0001] * 30:
            closes.append(closes[-1] + step)
        market = market_from_closes(closes)
        rsi_signals = generate_strategy_signals(market)[STRATEGIES[1]]
        features = build_features(market)

        self.assertEqual(list(rsi_signals["direction"]), ["long", "short"])
        for signal in rsi_signals.itertuples(index=False):
            index = int(signal.source_index)
            previous = features.loc[index - 1, "rsi_14"]
            current = features.loc[index, "rsi_14"]
            if signal.direction == "long":
                self.assertLessEqual(previous, 30.0)
                self.assertGreater(current, 30.0)
            else:
                self.assertGreaterEqual(previous, 70.0)
                self.assertLess(current, 70.0)

    def test_donchian_channel_excludes_signal_candle(self):
        closes = [1.1] * 20 + [1.1002] + [1.1002] * 8
        market = market_from_closes(closes, high_pad=0.0001, low_pad=0.0001)
        signals = generate_strategy_signals(market)[STRATEGIES[2]]

        # At index 20, the close clears the prior 20 highs. Its own high is
        # above its close, so including the current candle would hide the break.
        self.assertIn(20, signals["source_index"].tolist())
        feature_high = market.loc[:19, "high"].max()
        self.assertGreater(market.loc[20, "close"], feature_high)
        self.assertGreater(market.loc[20, "high"], market.loc[20, "close"])


class StrategyEvaluationTests(unittest.TestCase):
    def test_backtester_integration_keeps_outcome_inside_period(self):
        market = flat_market()
        period = timeline_periods(market)["train"]
        usable_index = period["end_index"] - 25
        boundary_index = period["end_index"] - 17
        signals = pd.DataFrame({
            "signal_id": [
                f"test:{market.loc[usable_index, 'time'].isoformat()}",
                f"test:{market.loc[boundary_index, 'time'].isoformat()}",
            ],
            "decision_timestamp": [
                market.loc[usable_index, "time"] + pd.Timedelta("1min"),
                market.loc[boundary_index, "time"] + pd.Timedelta("1min"),
            ],
            "direction": ["long", "short"],
            "source_row_index": [usable_index, boundary_index],
            "source_index": [usable_index, boundary_index],
            "source_candle_timestamp": [
                market.loc[usable_index, "time"],
                market.loc[boundary_index, "time"],
            ],
            "indicator_warmup_complete": [True, True],
        })
        result = evaluate_period(
            market, signals, period,
            execution_delay="0s", gap_tolerance="90s",
            transaction_cost_pips=1.0,
            expected_signal_id_prefix="test",
        )

        self.assertEqual(result["summary"]["number_of_trades"], 1)
        self.assertEqual(result["rejected"]["unavailable_exit"], 1)
        trade = result["trades"].iloc[0]
        self.assertGreater(trade.entry_timestamp, trade.decision_timestamp)
        self.assertEqual(trade.exit_timestamp, trade.entry_timestamp + pd.Timedelta("15min"))
        self.assertLess(trade.exit_timestamp, period["end"])
        self.assertEqual(trade.transaction_costs_pips, 1.0)
        self.assertEqual(trade.signal_id, f"test:{market.loc[usable_index, 'time'].isoformat()}")
        self.assertEqual(trade.source_row_index, usable_index)
        self.assertEqual(trade.source_candle_timestamp, market.loc[usable_index, "time"])
        rejected = result["rejected_signals"]
        self.assertEqual(
            rejected.loc[0, "signal_id"],
            f"test:{market.loc[boundary_index, 'time'].isoformat()}",
        )
        self.assertEqual(rejected.loc[0, "rejection_reason"], "unavailable_exit")
        self.assertEqual(result["generated_signals_in_period"], 2)
        self.assertEqual(
            result["generated_signals_in_period"],
            result["rejected_warmup"] + len(result["trades"])
            + sum(result["rejected"].values()),
        )
        self.assertEqual(result["accepted_attributed"], 1)
        self.assertEqual(result["rejected_attributed"], 1)
        rendered = render_report(
            timeline_periods(market),
            {
                name: {strategy: result for strategy in STRATEGIES}
                for name in ("train", "validation", "test")
            },
            market=market,
        )
        self.assertIn("Candidate events", rendered)
        self.assertIn("Pre-backtester warm-up exclusions", rendered)
        self.assertIn("Accepted trades", rendered)
        self.assertIn("Backtester rejected", rendered)
        self.assertIn("| EMA 9/21 crossover | 2 | 0 / 0 | 1 |", rendered)

    def test_warmup_candidate_is_suppressed_and_attributed(self):
        market = flat_market()
        period = timeline_periods(market)["train"]
        index = period["start_index"] + 110
        signals = pd.DataFrame({
            "signal_id": [
                f"test:{market.loc[index, 'time'].isoformat()}",
                f"test:{market.loc[index + 10, 'time'].isoformat()}",
            ],
            "decision_timestamp": [
                market.loc[index, "time"] + pd.Timedelta("1min"),
                market.loc[index + 10, "time"] + pd.Timedelta("1min"),
            ],
            "direction": ["long", "short"],
            "source_row_index": [index, index + 10],
            "source_candle_timestamp": [market.loc[index, "time"], market.loc[index + 10, "time"]],
            "indicator_warmup_complete": [False, True],
        })
        result = evaluate_period(
            market, signals, period, execution_delay="0s",
            gap_tolerance="90s", transaction_cost_pips=0.0,
            expected_signal_id_prefix="test",
        )

        self.assertEqual(result["summary"]["number_of_trades"], 1)
        self.assertEqual(
            result["trades"].iloc[0].signal_id,
            f"test:{market.loc[index + 10, 'time'].isoformat()}",
        )
        self.assertEqual(result["rejected_warmup"], 1)
        warmup_reject = result["rejected_signals"].loc[
            result["rejected_signals"]["rejection_reason"] == "indicator_warmup_initial"
        ].iloc[0]
        self.assertEqual(warmup_reject.signal_id, f"test:{market.loc[index, 'time'].isoformat()}")
        self.assertEqual(result["candidate_attribution_verified"], 2)

    def test_attribution_rejects_mismatched_row_timestamp(self):
        market = flat_market()
        records = pd.DataFrame({
            "signal_id": [f"ema_9_21:{market.loc[8, 'time'].isoformat()}"],
            "source_row_index": [7],
            "source_candle_timestamp": [market.loc[8, "time"]],
        })
        with self.assertRaisesRegex(ValueError, "row/timestamp mismatch"):
            validate_signal_attribution(
                records, market, expected_signal_id_prefix="ema_9_21"
            )

    def test_attribution_rejects_malformed_or_wrong_strategy_id(self):
        market = flat_market()
        for signal_id in (
            "not-an-id",
            f"rsi_14:{market.loc[7, 'time'].isoformat()}",
            "ema_9_21:2025-01-01T00:07:00",
        ):
            with self.subTest(signal_id=signal_id):
                records = pd.DataFrame({
                    "signal_id": [signal_id],
                    "source_row_index": [7],
                    "source_candle_timestamp": [market.loc[7, "time"]],
                })
                with self.assertRaisesRegex(ValueError, "signal_id"):
                    validate_signal_attribution(
                        records, market, expected_signal_id_prefix="ema_9_21"
                    )

    def test_attribution_without_source_fields_is_counted_unavailable(self):
        verified, unavailable = validate_signal_attribution(
            pd.DataFrame({"signal_id": ["unknown"]}), flat_market(),
            expected_signal_id_prefix="ema_9_21",
        )
        self.assertEqual((verified, unavailable), (0, 1))


if __name__ == "__main__":
    unittest.main()
