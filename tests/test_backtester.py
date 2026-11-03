"""Synthetic tests for the sequential OHLC backtesting engine."""

import unittest

import numpy as np
import pandas as pd

from src.research.backtester import run_backtest


def make_market(size=130):
    times = pd.date_range("2025-01-01", periods=size, freq="min", tz="UTC")
    opens = np.full(size, 1.1)
    closes = np.full(size, 1.1)
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": np.full(size, 1.2),
        "low": np.full(size, 1.0),
        "close": closes,
    })


def signal_at(market, row_index, direction):
    return {
        "decision_timestamp": market.loc[row_index, "time"] + pd.Timedelta("1min"),
        "direction": direction,
    }


def execute(market, signals, **kwargs):
    options = {
        "holding_period": "2min",
        "timestamp_convention": "bar_open",
        "transaction_cost_pips": 0.0,
        "gap_tolerance": "90s",
    }
    options.update(kwargs)
    return run_backtest(pd.DataFrame(signals), market, **options)


class BacktesterTests(unittest.TestCase):
    def test_candidates_are_processed_chronologically(self):
        market = make_market()
        result = execute(market, [
            signal_at(market, 90, "short"),
            signal_at(market, 70, "long"),
            signal_at(market, 80, "short"),
        ], allow_overlapping=True)
        trades = result["trades"]

        self.assertEqual(len(trades), 3)
        self.assertTrue(trades["decision_timestamp"].is_monotonic_increasing)
        self.assertTrue((trades["entry_timestamp"] > trades["decision_timestamp"]).all())

    def test_default_skips_overlapping_positions(self):
        market = make_market()
        result = execute(market, [
            signal_at(market, 70, "long"),
            signal_at(market, 71, "short"),
            signal_at(market, 90, "long"),
        ])

        self.assertEqual(result["summary"]["number_of_trades"], 2)
        self.assertEqual(result["rejected"]["overlap_skipped"], 1)
        trades = result["trades"]
        self.assertGreater(trades.iloc[1].entry_timestamp, trades.iloc[0].exit_timestamp)

    def test_long_short_pnl_and_fixed_cost(self):
        market = make_market()
        market.loc[73, "close"] = 1.1003  # Long exits +3 pips from 1.1000.
        market.loc[83, "close"] = 1.0998  # Short exits +2 pips from 1.1000.
        result = execute(
            market,
            [signal_at(market, 70, "long"), signal_at(market, 80, "short")],
            transaction_cost_pips=0.5,
        )
        trades = result["trades"]

        self.assertEqual(list(trades["direction"]), ["long", "short"])
        self.assertAlmostEqual(trades.iloc[0].gross_pnl_pips, 3.0)
        self.assertAlmostEqual(trades.iloc[1].gross_pnl_pips, 2.0)
        self.assertTrue((trades["transaction_costs_pips"] == 0.5).all())
        self.assertAlmostEqual(trades.iloc[0].net_pnl_pips, 2.5)
        self.assertAlmostEqual(trades.iloc[1].net_pnl_pips, 1.5)

    def test_missing_entry_or_exit_prices_reject_trades(self):
        market = make_market()
        market.loc[72, "open"] = np.nan
        market.loc[83, "close"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing.*OHLC"):
            execute(market, [
                signal_at(market, 70, "long"),
                signal_at(market, 80, "short"),
            ])

        result = execute(make_market(), [
            {"decision_timestamp": pd.NaT, "direction": "long"},
        ])
        self.assertEqual(result["rejected"]["invalid_decision_timestamp"], 1)

    def test_nonfinite_and_nonpositive_ohlc_values_are_rejected(self):
        for column in ("open", "high", "low", "close"):
            for invalid_price in (np.nan, np.inf, 0.0, -1.0):
                with self.subTest(column=column, invalid_price=invalid_price):
                    market = make_market()
                    market.loc[75, column] = invalid_price
                    with self.assertRaisesRegex(ValueError, "malformed|inconsistent|OHLC"):
                        execute(market, [signal_at(market, 70, "long")])

    def test_summary_break_even_profit_factor_and_drawdown(self):
        market = make_market()
        # Net trade returns at cost=1 pip: +2, -1, 0, +3.
        market.loc[73, "close"] = 1.1003
        market.loc[83, "close"] = 1.1000
        market.loc[93, "close"] = 1.1001
        market.loc[103, "close"] = 1.1004
        result = execute(
            market,
            [
                signal_at(market, 70, "long"),
                signal_at(market, 80, "short"),
                signal_at(market, 90, "long"),
                signal_at(market, 100, "long"),
            ],
            transaction_cost_pips=1.0,
        )
        summary = result["summary"]

        self.assertEqual(summary["number_of_trades"], 4)
        self.assertEqual(summary["win_rate"], 0.5)  # Break-even remains in denominator.
        self.assertEqual(summary["break_even_trades"], 1)
        self.assertAlmostEqual(summary["total_net_pips"], 4.0)
        self.assertAlmostEqual(summary["average_net_pips"], 1.0)
        self.assertAlmostEqual(summary["profit_factor"], 5.0)
        self.assertAlmostEqual(summary["max_drawdown_pips"], 1.0)

    def test_explicitly_allowing_overlap_keeps_both_signals(self):
        market = make_market()
        result = execute(market, [
            signal_at(market, 70, "long"),
            signal_at(market, 71, "short"),
        ], allow_overlapping=True)

        self.assertEqual(result["summary"]["number_of_trades"], 2)
        self.assertEqual(result["rejected"]["overlap_skipped"], 0)

    def test_incomplete_trade_at_end_of_data_is_rejected(self):
        market = make_market(size=30)
        result = execute(
            market,
            [signal_at(market, 26, "long")],
            holding_period="10min",
        )

        self.assertTrue(result["trades"].empty)
        self.assertEqual(result["rejected"]["unavailable_exit"], 1)

    def test_signals_before_across_and_after_gap(self):
        market = make_market(size=60).drop(index=30).reset_index(drop=True)
        signals = [
            {"decision_timestamp": pd.Timestamp("2025-01-01 00:25", tz="UTC"), "direction": "long"},
            {"decision_timestamp": pd.Timestamp("2025-01-01 00:29", tz="UTC"), "direction": "short"},
            {"decision_timestamp": pd.Timestamp("2025-01-01 00:32", tz="UTC"), "direction": "long"},
        ]
        result = execute(
            market,
            signals,
            holding_period="10min",
        )

        self.assertEqual(result["rejected"]["gap_exceeds_tolerance"], 2)
        self.assertEqual(result["summary"]["number_of_trades"], 1)
        self.assertEqual(result["trades"].iloc[0].decision_timestamp, signals[2]["decision_timestamp"])

    def test_simultaneous_exits_have_order_independent_drawdown(self):
        market = make_market(size=110)
        market.loc[73, "close"] = 1.1010  # Earlier +10 pip trade.
        market.loc[83, "close"] = 1.1005  # Same-time long/short gross +/-5.
        prior = signal_at(market, 70, "long")
        concurrent = [
            {"decision_timestamp": pd.Timestamp("2025-01-01 01:21", tz="UTC"), "direction": "long"},
            {"decision_timestamp": pd.Timestamp("2025-01-01 01:21", tz="UTC"), "direction": "short"},
        ]

        first_order = execute(
            market, [prior, *concurrent], allow_overlapping=True,
            transaction_cost_pips=1.0,
        )
        reverse_order = execute(
            market, [prior, *reversed(concurrent)], allow_overlapping=True,
            transaction_cost_pips=1.0,
        )

        self.assertEqual(first_order["trades"].iloc[1].exit_timestamp,
                         first_order["trades"].iloc[2].exit_timestamp)
        self.assertEqual(first_order["summary"]["max_drawdown_pips"], 2.0)
        self.assertEqual(
            first_order["summary"]["max_drawdown_pips"],
            reverse_order["summary"]["max_drawdown_pips"],
        )

    def test_non_utc_aware_timestamps_normalize_without_shifting_entry(self):
        market = make_market()
        market["time"] = market["time"].dt.tz_convert("Europe/Budapest")
        signal = signal_at(market, 70, "long")
        result = execute(market, [signal])
        trade = result["trades"].iloc[0]

        expected_entry = make_market().loc[72, "time"]
        self.assertEqual(str(trade.entry_timestamp.tz), "UTC")
        self.assertEqual(trade.entry_timestamp, expected_entry)
        self.assertGreater(trade.entry_timestamp, trade.decision_timestamp)

    def test_naive_custom_timestamps_are_rejected_as_ambiguous(self):
        market = make_market()
        market["time"] = market["time"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "Naive timestamps require"):
            execute(market, [signal_at(market, 70, "long")])


if __name__ == "__main__":
    unittest.main()
