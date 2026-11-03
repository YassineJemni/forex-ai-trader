"""Synthetic tests for the fixed EMA crossover experiment plumbing."""

import unittest

import numpy as np
import pandas as pd

from src.research.experiment_ema import (
    evaluate_period,
    generate_ema_crossover_signals,
    timeline_periods,
)


def market_from_closes(closes):
    closes = np.asarray(closes, dtype=float)
    times = pd.date_range("2025-01-01", periods=len(closes), freq="min", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": closes,
        "high": closes + 0.0001,
        "low": closes - 0.0001,
        "close": closes,
        "volume": np.ones(len(closes)),
    })


def flat_market(size=240):
    return market_from_closes(np.full(size, 1.1))


class EMASignalTests(unittest.TestCase):
    def test_crossover_is_decided_at_signal_candle_close(self):
        closes = [1.1] * 30 + [1.11] * 25 + [1.09] * 30
        market = market_from_closes(closes)
        signals = generate_ema_crossover_signals(market)

        self.assertEqual(list(signals["direction"]), ["long", "short"])
        for _, signal in signals.iterrows():
            source = market.iloc[int(signal.source_index)]
            self.assertEqual(signal.decision_timestamp, source.time + pd.Timedelta("1min"))
            self.assertGreater(signal.decision_timestamp, source.time)

    def test_future_prices_do_not_change_earlier_signals(self):
        closes = [1.1] * 30 + [1.11] * 25 + [1.09] * 30
        market = market_from_closes(closes)
        baseline = generate_ema_crossover_signals(market)
        cutoff = market.loc[45, "time"]
        altered = market.copy()
        altered.loc[46:, ["open", "high", "low", "close"]] += 0.05
        changed = generate_ema_crossover_signals(altered)

        early_baseline = baseline.loc[baseline["decision_timestamp"] <= cutoff].reset_index(drop=True)
        early_changed = changed.loc[changed["decision_timestamp"] <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(early_baseline, early_changed)


class EMABacktestIntegrationTests(unittest.TestCase):
    def test_trade_finishes_before_period_boundary_and_uses_shared_backtester(self):
        market = flat_market()
        periods = timeline_periods(market)
        period = periods["train"]
        before_boundary = period["end_index"] - 25
        crossing_boundary = period["end_index"] - 17
        signals = pd.DataFrame({
            "decision_timestamp": [
                market.loc[before_boundary, "time"] + pd.Timedelta("1min"),
                market.loc[crossing_boundary, "time"] + pd.Timedelta("1min"),
            ],
            "direction": ["long", "short"],
            "source_index": [before_boundary, crossing_boundary],
        })

        result = evaluate_period(
            market,
            signals,
            period,
            execution_delay="0s",
            gap_tolerance="90s",
            transaction_cost_pips=0.5,
        )

        self.assertEqual(result["summary"]["number_of_trades"], 1)
        self.assertEqual(result["rejected_at_period_end"], 1)
        trade = result["trades"].iloc[0]
        self.assertEqual(trade.exit_timestamp, trade.entry_timestamp + pd.Timedelta("15min"))
        self.assertGreater(trade.entry_timestamp, trade.decision_timestamp)
        self.assertLess(trade.exit_timestamp, period["end"])
        self.assertEqual(trade.transaction_costs_pips, 0.5)

    def test_default_positions_do_not_overlap(self):
        market = flat_market()
        period = timeline_periods(market)["train"]
        indices = [period["start_index"] + 110, period["start_index"] + 111]
        signals = pd.DataFrame({
            "decision_timestamp": [market.loc[i, "time"] + pd.Timedelta("1min") for i in indices],
            "direction": ["long", "short"],
            "source_index": indices,
        })
        result = evaluate_period(
            market, signals, period, execution_delay="0s",
            gap_tolerance="90s", transaction_cost_pips=0.0,
        )
        self.assertEqual(result["summary"]["number_of_trades"], 1)
        self.assertEqual(result["rejected"]["overlap_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
