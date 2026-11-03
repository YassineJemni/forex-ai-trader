import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research import mission13_3_validation as mission
from src.research.market_intelligence import load_pair_timeframe_validation
from src.research.multi_strategy_engine import strategy_library
from src.research.predictability import _holm_adjust


START = pd.Timestamp("2025-01-01T00:00:00Z")
STEP = {"M15": 15, "H1": 60, "H4": 240}


def all_results():
    return [
        {"strategy_id": strategy.strategy_id, "pair": pair, "timeframe": timeframe,
         "bars": 1000, "signals": 20}
        for strategy in strategy_library()
        for pair in mission.PAIRS
        for timeframe in mission.TIMEFRAMES
    ]


def all_boundaries():
    result = {}
    for pair in mission.PAIRS:
        for timeframe in mission.TIMEFRAMES:
            times = pd.date_range(START, periods=100, freq=f"{STEP[timeframe]}min")
            result[(pair, timeframe)] = (times[0], times[60], times[80],
                                          times[-1] + pd.Timedelta(microseconds=1))
    return result


def sample_trades(boundaries, *, count_per_period=10, gross=10.0):
    pair, timeframe = "EURUSD", "M15"
    strategy = mission.STRATEGY_IDS[0]
    start, validation_start, validation_end, _ = boundaries[(pair, timeframe)]
    rows = []
    for i in range(30):
        entry = start + pd.Timedelta(minutes=STEP[timeframe] * (i + 1))
        rows.append({
            "strategy_id": strategy, "pair": pair, "timeframe": timeframe,
            "decision_timestamp": entry - pd.Timedelta(minutes=1),
            "entry_timestamp": entry, "exit_timestamp": entry + pd.Timedelta(minutes=1),
            "direction": "BUY", "gross_pnl_pips": gross, "stop_distance_pips": 5.0,
            "regime": "trending", "session_utc": "London", "split": "train",
        })
    span = validation_end - validation_start
    for subperiod in range(mission.VALIDATION_SUBPERIODS):
        left = validation_start + span * subperiod / mission.VALIDATION_SUBPERIODS
        for index in range(count_per_period):
            entry = left + pd.Timedelta(minutes=index)
            rows.append({
                "strategy_id": strategy, "pair": pair, "timeframe": timeframe,
                "decision_timestamp": entry - pd.Timedelta(seconds=1),
                "entry_timestamp": entry, "exit_timestamp": entry + pd.Timedelta(seconds=1),
                "direction": "BUY", "gross_pnl_pips": gross, "stop_distance_pips": 5.0,
                "regime": "trending", "session_utc": "London", "split": "validation",
            })
    return pd.DataFrame(rows)


def test_full_exact_scope_preserves_all_294_strategy_pair_timeframe_cells():
    boundaries = all_boundaries()
    result = mission.evaluate_exact_scope(all_results(), sample_trades(boundaries), boundaries,
                                          n_resamples=100)
    assert result["cell_count"] == 294
    assert result["hypothesis_family_size"] == 294
    assert {row["strategy_id"] for row in result["cell_evidence"]} == set(mission.STRATEGY_IDS)
    assert {row["pair"] for row in result["cell_evidence"]} == set(mission.PAIRS)
    assert {row["timeframe"] for row in result["cell_evidence"]} == set(mission.TIMEFRAMES)
    exact = next(row for row in result["cell_evidence"]
                 if (row["strategy_id"], row["pair"], row["timeframe"])
                 == (mission.STRATEGY_IDS[0], "EURUSD", "M15"))
    assert exact["validation_metrics"]["trades"] == 30
    assert exact["validation_trades_per_30_calendar_days"] is not None
    assert exact["validation_signal_frequency_status"].startswith("unavailable")


def test_saved_train_validation_counts_are_reconciled():
    trades = sample_trades(all_boundaries())
    key = (mission.STRATEGY_IDS[0], "EURUSD", "M15")
    mission.validate_saved_split_counts([{
        "strategy_id": key[0], "pair": key[1], "timeframe": key[2],
        "splits": {"train": {"trades": 30}, "validation": {"trades": 30}},
    }], trades)
    with np.testing.assert_raises_regex(ValueError, "trade count mismatch"):
        mission.validate_saved_split_counts([{
            "strategy_id": key[0], "pair": key[1], "timeframe": key[2],
            "splits": {"train": {"trades": 29}, "validation": {"trades": 30}},
        }], trades)


def test_small_samples_are_not_bootstrapped_or_marked_stable():
    boundaries = all_boundaries()
    few = sample_trades(boundaries, count_per_period=2)
    result = mission.evaluate_exact_scope(all_results(), few, boundaries, n_resamples=100)
    cell = next(row for row in result["cell_evidence"] if row["strategy_id"] == mission.STRATEGY_IDS[0]
                and row["pair"] == "EURUSD" and row["timeframe"] == "M15")
    assert cell["validation_metrics"]["trades"] == 6
    assert cell["pair_timeframe_stability_status"] == "unstable_or_insufficient_validation"
    assert cell["classification"] == "INSUFFICIENT EVIDENCE"
    assert cell["validation_bootstrap"]["lower"] is None
    assert cell["validation_bootstrap"]["bootstrap_one_sided_p_value"] == 1


def test_bootstrap_interval_and_test_are_deterministic_and_dependence_aware():
    values = np.array([0.3, 0.4, -0.1, 0.5, 0.2] * 8, dtype=float)
    first = mission._bootstrap_mean(values, block_length=5, n_resamples=200, seed=19)
    second = mission._bootstrap_mean(values, block_length=5, n_resamples=200, seed=19)
    assert first == second
    assert first["lower"] < first["mean"] < first["upper"]
    assert 0 < first["bootstrap_one_sided_p_value"] <= 1
    assert first["method"] == "circular_moving_block_percentile_ci_and_centered_null_test"


def test_holm_correction_controls_the_fixed_cell_family():
    adjusted = _holm_adjust(np.array([0.02, 0.03, 0.5, 1.0]))
    assert adjusted.tolist() == [0.08, 0.09, 1.0, 1.0]
    assert not (adjusted <= 0.05).any()


def test_validation_stability_uses_three_chronological_periods():
    boundaries = all_boundaries()
    result = mission.evaluate_exact_scope(all_results(), sample_trades(boundaries), boundaries,
                                          n_resamples=100)
    cell = next(row for row in result["cell_evidence"]
                if row["strategy_id"] == mission.STRATEGY_IDS[0]
                and row["pair"] == "EURUSD" and row["timeframe"] == "M15")
    assert [row["trades"] for row in cell["validation_periods"]] == [10, 10, 10]
    assert cell["pair_timeframe_stability_status"] == "stable_pair_timeframe_validation"


def test_cost_sensitivity_keeps_spread_commission_and_slippage_explicit():
    boundaries = all_boundaries()
    trades = sample_trades(boundaries)
    result = mission.evaluate_exact_scope(all_results(), trades, boundaries, n_resamples=100)
    rows = [row for row in result["training_validation_metrics_by_cost"]
            if row["strategy_id"] == mission.STRATEGY_IDS[0] and row["pair"] == "EURUSD"
            and row["timeframe"] == "M15" and row["split"] == "validation"]
    assert len(rows) == 27
    zero = next(row for row in rows if row["scenario"] == "spread_0_commission_0_slippage_0_pips_rt")
    stress = next(row for row in rows if row["scenario"] == mission.STRESS_COST_SCENARIO)
    assert zero["mean_outcome_pips"] - stress["mean_outcome_pips"] == 3.5
    assert not zero["assumptions_are_broker_calibrated"]
    assert result["cost_scenarios"][-1]["broker_calibrated"] is False


def test_unknown_validation_correlation_remains_unknown():
    boundaries = all_boundaries()
    trades = sample_trades(boundaries)
    additional = trades.copy()
    additional["strategy_id"] = mission.STRATEGY_IDS[1]
    both = pd.concat([trades[trades.split == "validation"], additional[additional.split == "validation"]],
                     ignore_index=True)
    pairs = mission._pairwise_validation_complementarity(both, [], [], [])
    assert pairs
    assert all(row["classification"] == "unknown_or_insufficient_complementarity"
               for row in pairs)
    assert all(row["signal_directional_overlap_status"].startswith("unknown")
               for row in pairs)


def test_directional_trade_overlap_counts_same_and_opposing_sides():
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    rows = []
    for strategy, direction in (("a", "BUY"), ("b", "SELL")):
        rows.append({"strategy_id": strategy, "pair": "EURUSD", "timeframe": "M15",
                     "entry_timestamp": base, "exit_timestamp": base + pd.Timedelta(minutes=3),
                     "direction": direction})
    result = mission._directional_trade_overlap(pd.DataFrame(rows))
    assert result[0]["opposing_direction_overlapping_trade_pairs"] == 1
    assert result[0]["same_direction_overlapping_trade_pairs"] == 0


def test_report_writer_records_hash_provenance_and_skips_test_outcomes(tmp_path):
    data_dir = tmp_path / "market-data"
    data_dir.mkdir()
    provenance = []
    for pair in mission.PAIRS:
        for timeframe in mission.TIMEFRAMES:
            path = data_dir / f"{pair}_{timeframe}.parquet"
            times = pd.date_range(START, periods=100, freq=f"{STEP[timeframe]}min")
            pd.DataFrame({"time": times}).to_parquet(path, index=False)
            provenance.append({"source_file": str(path), "research_fingerprint_sha256": "synthetic"})

    m8_path = tmp_path / "mission8.json"
    trade_path = tmp_path / "trades.csv"
    m9_path = tmp_path / "mission9.json"
    m11_path = tmp_path / "mission11.json"
    payload = {"strategies": 14, "results": all_results(),
               "data_exclusions_and_provenance": provenance, "data_fingerprints": {}}
    m8_path.write_text(json.dumps(payload), encoding="utf-8")
    m9_path.write_text("{}", encoding="utf-8")
    m11_path.write_text("{}", encoding="utf-8")
    rows = sample_trades(all_boundaries()).to_dict(orient="records")
    rows.append({
        "strategy_id": mission.STRATEGY_IDS[0], "pair": "EURUSD", "timeframe": "M15",
        "decision_timestamp": "2025-01-02T00:00:00Z",
        "entry_timestamp": "2025-01-02T00:00:00Z",
        "exit_timestamp": "2025-01-02T00:01:00Z", "direction": "BUY",
        "gross_pnl_pips": 999999, "stop_distance_pips": 1, "regime": "test_only",
        "session_utc": "test_only", "split": "test",
    })
    fields = list(rows[0])
    with trade_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = mission.run_mission13_3(
        mission8_path=m8_path, trade_path=trade_path, mission9_path=m9_path,
        mission11_path=m11_path, data_dir=data_dir, output_dir=tmp_path / "reports",
        report_stamp="20261103T120000Z", n_resamples=100)
    assert report["summary"]["exact_pair_timeframe_cells"] == 294
    assert report["trade_sample_accounting"]["post_validation_rows_skipped"] == 1
    assert report["software_provenance"]["test_outcomes_used"] is False
    assert len(report["source_fingerprints_sha256"]) >= 3 + len(mission.PAIRS) * len(mission.TIMEFRAMES)
    assert report["source_fingerprints_sha256"]["mission9_cost_report_reference"] == hashlib.sha256(
        m9_path.read_bytes()).hexdigest()
    assert all(Path(path).exists() for path in report["report_files"].values())


def test_all_fourteen_existing_engines_remain_unchanged():
    assert len(strategy_library()) == 14
    assert tuple(row.strategy_id for row in strategy_library()) == mission.STRATEGY_IDS


def test_mission13_2_gate_loads_only_fingerprint_matched_stability(tmp_path):
    mission9_path = tmp_path / "mission9.json"
    mission9_path.write_text(json.dumps({
        "metrics": [{
            "strategy_id": "ema_crossover", "pair": "EURUSD", "timeframe": "M15",
            "split": "validation", "scenario": mission.STRESS_COST_SCENARIO,
            "trades": 30, "expectancy_r": 0.1,
        }],
        "market_regime": [],
    }), encoding="utf-8")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "source_fingerprints_sha256": {
            "mission9_cost_report_reference": hashlib.sha256(mission9_path.read_bytes()).hexdigest(),
        },
        "cells": [{
            "strategy_id": "ema_crossover", "pair": "EURUSD", "timeframe": "M15",
            "pair_timeframe_stability_status": "stable_pair_timeframe_validation",
        }],
    }), encoding="utf-8")
    loaded = load_pair_timeframe_validation(mission9_path, stability_path=gate_path)
    assert loaded[("ema_crossover", "EURUSD", "M15")]["pair_timeframe_stability_status"] == \
        "stable_pair_timeframe_validation"
