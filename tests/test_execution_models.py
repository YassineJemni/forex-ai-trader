"""Synthetic tests for opt-in side-aware execution and run identity."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.research.backtester import REJECTION_REASONS, TRADE_COLUMNS, run_backtest
from src.research.execution_models import (
    BidAskExecutionModel,
    MarketDataMetadata,
    prepare_bid_ask_inputs,
)
from src.research.experiment_store import ExperimentStore
from src.research.strategy_registry import get_strategy


def market(size: int = 12, *, ask_spread: float = 0.0002) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("2025-01-01", periods=size, freq="min", tz="UTC")
    bid_open = np.full(size, 1.1000)
    bid_close = np.full(size, 1.1000)
    bid_close[4] = 1.1010
    ask_open = bid_open + ask_spread
    ask_close = bid_close + ask_spread
    bid = pd.DataFrame({
        "time": times, "open": bid_open, "high": np.maximum(bid_open, bid_close) + 0.0001,
        "low": np.minimum(bid_open, bid_close) - 0.0001, "close": bid_close,
    })
    ask = pd.DataFrame({
        "time": times, "open": ask_open, "high": np.maximum(ask_open, ask_close) + 0.0001,
        "low": np.minimum(ask_open, ask_close) - 0.0001, "close": ask_close,
    })
    return bid, ask


def metadata(side: str, **overrides) -> MarketDataMetadata:
    return MarketDataMetadata(
        side=side, symbol=overrides.get("symbol", "EUR/USD"),
        timeframe=overrides.get("timeframe", "M1"),
        timestamp_convention=overrides.get("timestamp_convention", "bar_open"),
        timezone=overrides.get("timezone", "UTC"),
        provider=overrides.get("provider", "synthetic fixture"),
    )


def signal(direction: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "signal_id": f"fixture:{direction}:2", "source_row_index": 2,
        "source_candle_timestamp": pd.Timestamp("2025-01-01 00:02", tz="UTC"),
        "decision_timestamp": pd.Timestamp("2025-01-01 00:02", tz="UTC"),
        "direction": direction,
    }])


def execute(bid, ask, direction="long", **overrides):
    bid_meta = overrides.pop("bid_metadata", metadata("BID"))
    ask_meta = overrides.pop("ask_metadata", metadata("ASK"))
    model = overrides.pop("execution_model", BidAskExecutionModel(0.5, 0.25))
    return run_backtest(
        signal(direction), bid, ask_data=ask, bid_metadata=bid_meta,
        ask_metadata=ask_meta, execution_model=model,
        holding_period="2min", timestamp_convention="bar_open", bar_duration="1min",
        pip_size=0.0001, **overrides,
    )


class SideAwareExecutionTests(unittest.TestCase):
    def test_long_short_fills_include_spread_once_and_apply_explicit_costs(self):
        bid, ask = market()
        long = execute(bid, ask, "long")["trades"].iloc[0]
        short = execute(bid, ask, "short")["trades"].iloc[0]

        self.assertEqual(long.entry_price_side, "ASK")
        self.assertEqual(long.exit_price_side, "BID")
        self.assertAlmostEqual(long.entry_price, 1.1002)
        self.assertAlmostEqual(long.exit_price, 1.1010)
        self.assertAlmostEqual(long.mid_price_pnl_pips, 10.0)
        self.assertAlmostEqual(long.spread_cost_pips, 2.0)
        self.assertAlmostEqual(long.gross_pnl_pips, 8.0)
        self.assertAlmostEqual(long.commission_costs_pips, 0.5)
        self.assertAlmostEqual(long.slippage_costs_pips, 0.25)
        self.assertAlmostEqual(long.transaction_costs_pips, 0.75)
        self.assertAlmostEqual(long.net_pnl_pips, 7.25)

        self.assertEqual(short.entry_price_side, "BID")
        self.assertEqual(short.exit_price_side, "ASK")
        self.assertAlmostEqual(short.entry_price, 1.1000)
        self.assertAlmostEqual(short.exit_price, 1.1012)
        self.assertAlmostEqual(short.mid_price_pnl_pips, -10.0)
        self.assertAlmostEqual(short.spread_cost_pips, 2.0)
        self.assertAlmostEqual(short.gross_pnl_pips, -12.0)
        self.assertAlmostEqual(short.net_pnl_pips, -12.75)

    def test_zero_spread_matches_legacy_price_pnl_without_costs(self):
        bid, ask = market(ask_spread=0.0)
        no_cost_model = BidAskExecutionModel(0.0, 0.0)
        side_aware = execute(bid, ask, execution_model=no_cost_model)["trades"].iloc[0]
        legacy = run_backtest(
            signal("long"), bid, holding_period="2min", timestamp_convention="bar_open",
            bar_duration="1min", transaction_cost_pips=0.0,
        )["trades"].iloc[0]
        self.assertAlmostEqual(side_aware.entry_price, legacy.entry_price)
        self.assertAlmostEqual(side_aware.exit_price, legacy.exit_price)
        self.assertAlmostEqual(side_aware.gross_pnl_pips, legacy.gross_pnl_pips)
        self.assertAlmostEqual(side_aware.net_pnl_pips, legacy.net_pnl_pips)

    def test_strict_entry_and_elapsed_exit_timing_are_unchanged(self):
        bid, ask = market()
        trade = execute(bid, ask)["trades"].iloc[0]
        self.assertGreater(trade.entry_timestamp, trade.decision_timestamp)
        self.assertEqual(trade.entry_timestamp, bid.loc[3, "time"])
        # Entry at 00:03 plus two minutes exits at the close of the bar ending 00:05.
        self.assertEqual(trade.exit_timestamp, pd.Timestamp("2025-01-01 00:05", tz="UTC"))
        self.assertAlmostEqual(trade.exit_price, ask.loc[4, "close"] - 0.0002)

    def test_missing_exact_entry_or_exit_quote_is_rejected_with_attribution(self):
        bid, ask = market()
        missing_entry = ask.loc[ask.time != bid.loc[3, "time"]].reset_index(drop=True)
        result = execute(bid, missing_entry)
        self.assertEqual(result["rejected"]["unavailable_entry_quote"], 1)
        rejected = result["rejected_signals"].iloc[0]
        self.assertEqual(rejected.signal_id, "fixture:long:2")
        self.assertEqual(rejected.source_row_index, 2)

        missing_exit = ask.loc[ask.time != bid.loc[4, "time"]].reset_index(drop=True)
        result = execute(bid, missing_exit)
        self.assertEqual(result["rejected"]["unavailable_exit_quote"], 1)
        self.assertEqual(len(result["rejected_signals"]), 1)

    def test_shifted_timestamps_and_incompatible_source_metadata_are_rejected(self):
        bid, ask = market()
        shifted = ask.copy()
        shifted["time"] += pd.Timedelta("30s")
        with self.assertRaisesRegex(ValueError, "whole UTC minutes|M1 cadence"):
            execute(bid, shifted)

        with self.assertRaisesRegex(ValueError, "timeframe"):
            execute(bid, ask, ask_metadata=metadata("ASK", timeframe="M5"))
        with self.assertRaisesRegex(ValueError, "symbol"):
            execute(bid, ask, ask_metadata=metadata("ASK", symbol="GBP/USD"))
        with self.assertRaisesRegex(ValueError, "timestamp convention"):
            execute(bid, ask, ask_metadata=metadata("ASK", timestamp_convention="bar_close"))

    def test_negative_spread_is_a_signal_rejection(self):
        bid, ask = market()
        ask.loc[3, ["open", "high", "low", "close"]] -= 0.0003
        result = execute(bid, ask)
        self.assertEqual(result["rejected"]["invalid_entry_spread"], 1)
        self.assertEqual(result["rejected_signals"].iloc[0].signal_id, "fixture:long:2")

    def test_gap_rule_and_legacy_cost_rejection_are_explicit(self):
        bid, ask = market()
        gap_time = pd.Timestamp("2025-01-01 00:04", tz="UTC")
        bid.loc[4:, "time"] += pd.Timedelta("1min")
        ask.loc[4:, "time"] += pd.Timedelta("1min")
        result = execute(bid, ask)
        self.assertEqual(result["rejected"]["gap_exceeds_tolerance"], 1)

        with self.assertRaisesRegex(ValueError, "must be zero"):
            execute(bid, ask, transaction_cost_pips=1.0)

    def test_pairing_reports_unmatched_quotes_and_rejects_bad_ohlc(self):
        bid, ask = market()
        missing = ask.loc[ask.time != bid.loc[3, "time"]].reset_index(drop=True)
        _, _, report = prepare_bid_ask_inputs(
            bid, missing, bid_metadata=metadata("BID"), ask_metadata=metadata("ASK"),
            timestamp_convention="bar_open", bar_duration="1min",
        )
        self.assertEqual(report["bid_without_ask_timestamp"], 1)
        self.assertFalse(report["timestamps_are_exactly_aligned"])
        malformed = ask.copy()
        malformed.loc[2, "high"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            execute(bid, malformed)

    def test_identity_separates_legacy_fixed_cost_from_side_aware_cost_model(self):
        bid_meta, ask_meta = metadata("BID"), metadata("ASK")
        model = BidAskExecutionModel(0.5, 0.25)
        bid_fp, ask_fp = "a" * 64, "b" * 64
        side_identity = model.identity_payload(
            bid_metadata=bid_meta, ask_metadata=ask_meta,
            bid_fingerprint=bid_fp, ask_fingerprint=ask_fp,
            alignment_report={"matched_timestamps": 10},
        )
        base = {
            "strategy": get_strategy("ema_crossover"),
            "parameters": {"fast": 9, "slow": 21},
            "dataset_fingerprint": bid_fp,
            "dataset_identity": {"filename": "synthetic"},
            "date_start_utc": "2025-01-01T00:00:00Z",
            "date_end_utc": "2025-01-02T00:00:00Z",
            "execution_assumptions": {
                "experiment_kind": "unit_fixture", "transaction_cost_pips_round_trip": 1.0,
            },
            "gap_policy": {}, "software": {}, "split_metadata": {},
        }
        legacy_payload = ExperimentStore._request_payload(**base)
        side_payload = ExperimentStore._request_payload(**{
            **base,
            "execution_assumptions": {
                "experiment_kind": "unit_fixture", "transaction_cost_pips_round_trip": 0.0,
                "execution_model": side_identity,
            },
        })
        self.assertNotEqual(ExperimentStore._idempotency_key(legacy_payload),
                            ExperimentStore._idempotency_key(side_payload))
        self.assertFalse(side_identity["cost_components"]["legacy_fixed_round_trip_cost_applied"])

        higher_commission = BidAskExecutionModel(0.75, 0.25).identity_payload(
            bid_metadata=bid_meta, ask_metadata=ask_meta,
            bid_fingerprint=bid_fp, ask_fingerprint=ask_fp,
            alignment_report={"matched_timestamps": 10},
        )
        changed_data = BidAskExecutionModel(0.5, 0.25).identity_payload(
            bid_metadata=bid_meta, ask_metadata=ask_meta,
            bid_fingerprint=bid_fp, ask_fingerprint="d" * 64,
            alignment_report={"matched_timestamps": 10},
        )
        for changed in (higher_commission, changed_data):
            payload = ExperimentStore._request_payload(**{
                **base,
                "execution_assumptions": {
                    "experiment_kind": "unit_fixture", "transaction_cost_pips_round_trip": 0.0,
                    "execution_model": changed,
                },
            })
            self.assertNotEqual(ExperimentStore._idempotency_key(side_payload),
                                ExperimentStore._idempotency_key(payload))

    def test_side_cost_components_are_persistable_without_double_counting_spread(self):
        bid, ask = market()
        trades = execute(bid, ask)["trades"]
        result = ExperimentStore._period_payload({"trades": trades}, "train")
        costs = json.loads(result["costs"])
        self.assertAlmostEqual(costs["spread_total_pips_embedded_in_fill_prices"], 2.0)
        self.assertAlmostEqual(costs["commission_assumption_total_pips"], 0.5)
        self.assertAlmostEqual(costs["slippage_assumption_total_pips"], 0.25)
        self.assertAlmostEqual(costs["total_assumed_cost_pips"], 0.75)

    def test_legacy_default_output_contract_remains_unchanged(self):
        bid, _ = market()
        result = run_backtest(
            signal("long"), bid, holding_period="2min", timestamp_convention="bar_open",
            bar_duration="1min", transaction_cost_pips=1.0,
        )
        self.assertEqual(list(result["trades"].columns), TRADE_COLUMNS)
        self.assertEqual(tuple(result["rejected"].keys()), REJECTION_REASONS)
        self.assertEqual(result["trades"].iloc[0].transaction_costs_pips, 1.0)
        self.assertNotIn("execution_model", result)


if __name__ == "__main__":
    unittest.main()
