import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.research.mission9_cost_research import (
    CostScenario, _complementarity, _metrics, _trade_overlap, analyze_saved_results,
    collect_spread_snapshot,
)


def test_cost_metrics_are_normalized_in_r_and_handle_empty_losses():
    frame = pd.DataFrame({"pnl": [2.0, -1.0], "stop_distance_pips": [2.0, 2.0],
                          "gross_pnl_pips": [2.0, -1.0]})
    result = _metrics(frame, "pnl")
    assert result["expectancy_r"] == 0.25
    assert result["profit_factor"] == 2.0
    assert result["max_drawdown_r"] == 0.5
    assert result["max_consecutive_losses"] == 1


def test_overlap_counts_closed_interval_intersections():
    frame = pd.DataFrame([
        {"pair": "EURUSD", "timeframe": "H1", "split": "validation", "strategy_id": "a",
         "entry_timestamp": pd.Timestamp("2025-01-01T00:00Z"), "exit_timestamp": pd.Timestamp("2025-01-01T03:00Z")},
        {"pair": "EURUSD", "timeframe": "H1", "split": "validation", "strategy_id": "b",
         "entry_timestamp": pd.Timestamp("2025-01-01T02:00Z"), "exit_timestamp": pd.Timestamp("2025-01-01T04:00Z")},
        {"pair": "EURUSD", "timeframe": "H1", "split": "validation", "strategy_id": "b",
         "entry_timestamp": pd.Timestamp("2025-01-01T05:00Z"), "exit_timestamp": pd.Timestamp("2025-01-01T06:00Z")},
    ])
    assert _trade_overlap(frame)[0]["overlapping_trade_pairs"] == 1


def test_complementarity_uses_agreement_overlap_and_validation_correlation():
    relationships = [{"pair": f"EURUSD", "timeframe": tf, "strategy_a": "a", "strategy_b": "b",
                      "coincident_timestamps": 10,
                      "directional_agreement_when_coincident": 0.4,
                      "interpretation": "mixed"} for tf in ("M15", "H1", "H4")]
    overlaps = [{"pair": "EURUSD", "timeframe": tf, "split": "validation",
                 "strategy_a": "a", "strategy_b": "b", "overlapping_trade_pairs": 0,
                 "trades_a": 10, "trades_b": 10} for tf in ("M15", "H1", "H4")]
    correlations = [{"pair": "EURUSD", "timeframe": tf, "split": "validation",
                     "scenario": "stress", "strategy_a": "a", "strategy_b": "b",
                     "daily_exit_return_correlation": 0.1, "observed_exit_days": 100}
                    for tf in ("M15", "H1", "H4")]
    result = _complementarity(relationships, overlaps, correlations, "stress")
    assert result[0]["descriptive_interpretation"] == "potentially_complementary; validate further"
    assert result[0]["test_period_used"] is False


def test_saved_results_reuse_split_counts_and_separate_holdout(tmp_path):
    meta = {"generated_at_utc": "2026-11-03T00:00:00Z", "strategies": ["s"],
            "results": [{"strategy_id": "s", "pair": "EURUSD", "timeframe": "M15",
                         "splits": {"train": {"trades": 1}, "validation": {"trades": 1},
                                    "test": {"trades": 1}}}],
            "signal_relationships": [{"pair": "EURUSD", "timeframe": "M15",
                                      "strategy_a": "s", "strategy_b": "s2",
                                      "coincident_timestamps": 2,
                                      "directional_agreement_when_coincident": 0.5,
                                      "interpretation": "mixed"}]}
    meta_path = tmp_path / "mission8.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    rows = []
    for split_i, gross in enumerate((3.0, 1.0, 100.0)):
        rows.append({"strategy_id": "s", "pair": "EURUSD", "timeframe": "M15",
                     "entry_timestamp": f"2025-01-0{split_i + 1}T00:00:00Z",
                     "exit_timestamp": f"2025-01-0{split_i + 1}T01:00:00Z", "direction": "BUY",
                     "gross_pnl_pips": gross, "stop_distance_pips": 2.0,
                     "regime": "ranging", "session_utc": "London"})
    trade_path = tmp_path / "trades.csv"
    pd.DataFrame(rows).to_csv(trade_path, index=False)
    result = analyze_saved_results(meta_path, trade_path, [CostScenario(0.5, 0.25, 0.25)])
    net_validation = next(x for x in result["metrics"]
                          if x["split"] == "validation" and x["scenario"] != "gross_before_costs")
    assert net_validation["expectancy_r"] == 0.0
    assert next(x for x in result["metrics"] if x["split"] == "test" and
                x["scenario"] == "gross_before_costs")["expectancy_r"] == 50.0
    assert result["candidate_assessment_validation_only"][0]["validation_trade_count"] == 1
    assert result["signal_relationship_summary"][0]["conflict_fraction_weighted"] == 0.5


def test_spread_snapshot_preserves_invalid_future_timestamp_status():
    class Info:
        spread = 1
        point = 0.00001
        digits = 5

        def _asdict(self):
            return {"spread": 1, "point": 0.00001}

    class Tick:
        bid, ask = 1.1, 1.1001
        time = 1_900_000_000
        time_msc = 1_900_000_000_000

    class Terminal:
        @staticmethod
        def symbol_info_tick(_): return Tick()
        @staticmethod
        def symbol_info(_): return Info()
        @staticmethod
        def last_error(): return (0, "")

    class Adapter:
        max_quote_age_seconds = 90
        server_timezone = "UTC"
        def connect(self): pass
        def resolve_symbol(self, symbol): return symbol
        def raw_tick(self, symbol): return Tick()
        def symbol_info(self, symbol): return Info()
        def normalize_tick_timestamp(self, tick, symbol):
            return datetime.fromtimestamp(tick.time, timezone.utc)
        def quote_from_tick(self, symbol, tick): raise RuntimeError("future quote")

    result = collect_spread_snapshot(Adapter(), ["EURUSD"])
    assert result["rows"][0]["spread_pips"] == pytest.approx(1.0)
    assert result["rows"][0]["timestamp_status"] == "future"
    assert result["distributions"][0]["valid_timestamp_samples"] == 0
