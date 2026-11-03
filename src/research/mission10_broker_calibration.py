"""Read-only broker cost calibration and saved-strategy diagnostics.

No strategy signals are regenerated. Quotes are sampled only through the
existing MT5 data adapter; original research data and reports are read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.research.mission9_cost_research import _assign_splits, _research_split_bounds
from src.trader.config import AppConfig, DEFAULT_PAIRS
from src.trader.market_sessions import DEFAULT_SESSION_WINDOWS, classify_utc_sessions, parse_session_windows
from src.trader.mt5_adapter import MT5DataAdapter, MT5Error


ROOT = Path(__file__).resolve().parents[2]
SESSION_NAMES = ("Asia", "London", "NewYork", "LondonNewYorkOverlap")
RELIABLE_SAMPLE_MIN = 30
RELIABLE_COVERAGE_MINUTES = 60
MISSION9_STRESS_SCENARIO = "spread_2_commission_1_slippage_0.5_pips_rt"


def pip_size(pair: str) -> float:
    compact = "".join(ch for ch in pair.upper() if ch.isalpha())
    if len(compact) < 6:
        raise ValueError(f"Cannot determine pip size for symbol {pair!r}")
    return 0.01 if compact[3:6] == "JPY" else 0.0001


def _spec_value(info: Any, name: str) -> Any:
    value = getattr(info, name, None)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def symbol_specification(adapter: MT5DataAdapter, requested: str) -> dict[str, Any]:
    broker = adapter.resolve_symbol(requested)
    info = adapter.symbol_info(broker)
    fields = ("trade_contract_size", "trade_tick_size", "trade_tick_value",
              "trade_tick_value_profit", "trade_tick_value_loss", "point", "digits",
              "volume_min", "volume_max", "volume_step", "trade_exemode",
              "trade_mode", "trade_calc_mode", "swap_long", "swap_short", "swap_mode",
              "spread", "spread_float")
    result = {"pair": requested, "broker_symbol": broker, "pip_size": pip_size(requested)}
    result.update({key: _spec_value(info, key) for key in fields})
    mapping = info._asdict() if hasattr(info, "_asdict") else {}
    result["commission_fields"] = {key: value for key, value in mapping.items()
                                   if "commission" in key.lower()}
    result["commission_schedule_available"] = bool(result["commission_fields"])
    return result


def _quote_observation(adapter: MT5DataAdapter, pair: str, observation_utc: datetime,
                       windows: dict[str, tuple[str, str]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    broker_symbol = adapter.resolve_symbol(pair)
    tick = adapter.raw_tick(broker_symbol)
    info = adapter.symbol_info(broker_symbol)
    raw_seconds = int(getattr(tick, "time", 0) or 0)
    raw_millis = int(getattr(tick, "time_msc", 0) or 0)
    bid = float(getattr(tick, "bid", float("nan")))
    ask = float(getattr(tick, "ask", float("nan")))
    normalized: datetime | None = None
    validation_status = "valid"
    validation_error = None
    freshness = None
    try:
        normalized = adapter.normalize_tick_timestamp(tick, broker_symbol)
        freshness = (observation_utc - normalized).total_seconds()
        quote = adapter.quote_from_tick(broker_symbol, tick)
        bid, ask = quote.bid, quote.ask
    except MT5Error as exc:
        validation_error = str(exc)
        low = validation_error.lower()
        validation_status = "future" if "future" in low else "stale" if "stale" in low else "invalid_quote"
    except (TypeError, ValueError, OverflowError) as exc:
        validation_status, validation_error = "invalid_timestamp", str(exc)
    if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
        validation_status, validation_error = "invalid_quote", "bid/ask must be finite, positive, and uncrossed"
    session_tags = classify_utc_sessions(observation_utc, windows)
    pip = pip_size(pair)
    raw_as_utc = (datetime.fromtimestamp(raw_millis / 1000, timezone.utc).isoformat()
                  if raw_millis else datetime.fromtimestamp(raw_seconds, timezone.utc).isoformat()
                  if raw_seconds else None)
    row = {"pair": pair, "broker_symbol": broker_symbol,
           "observation_utc": observation_utc.isoformat(),
           "raw_tick_time_seconds": raw_seconds or None,
           "raw_tick_time_milliseconds": raw_millis or None,
           "raw_timestamp_if_interpreted_as_utc": raw_as_utc,
           "normalized_tick_utc": normalized.isoformat() if normalized else None,
           "timezone_configuration": adapter.server_timezone,
           "session_tags_utc": session_tags,
           "primary_session_utc": next((name for name in session_tags if name != "LondonNewYorkOverlap"), None),
           "freshness_seconds": freshness,
           "quote_freshness": "fresh" if validation_status == "valid" else validation_status,
           "validation_status": validation_status, "validation_error": validation_error,
           "bid": bid if math.isfinite(bid) else None,
           "ask": ask if math.isfinite(ask) else None,
           "spread_price": ask - bid if math.isfinite(bid) and math.isfinite(ask) else None,
           "spread_pips": (ask - bid) / pip if math.isfinite(bid) and math.isfinite(ask) else None,
           "pip_size": pip, "symbol_point": _spec_value(info, "point"),
           "symbol_digits": _spec_value(info, "digits"),
           "symbol_spread_points": _spec_value(info, "spread"),
           "sample_duplicate": False}
    return row, symbol_specification(adapter, pair)


def _observation_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if not row.get("raw_tick_time_milliseconds"):
        return None
    return (row.get("pair"), row.get("broker_symbol"), row.get("raw_tick_time_milliseconds"),
            row.get("bid"), row.get("ask"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed observation JSONL line {line_number}: {exc}") from exc
    return rows


def collect_spreads(adapter: MT5DataAdapter, pairs: Iterable[str], *, duration_minutes: float,
                    interval_seconds: float, output_path: Path, checkpoint_path: Path,
                    resume: bool = False,
                    windows: dict[str, tuple[str, str]] | None = None,
                    progress=print) -> dict[str, Any]:
    """Bounded sampler; every row is appended immediately and every sweep checkpointed."""
    if duration_minutes <= 0 or duration_minutes > 24 * 60:
        raise ValueError("duration_minutes must be in (0, 1440]")
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be at least 1 second")
    pairs = tuple(pair for pair in pairs if pair)
    if not pairs:
        raise ValueError("At least one pair is required")
    if resume and (not output_path.exists() or not checkpoint_path.exists()):
        raise FileNotFoundError("--resume requires both the observation JSONL and checkpoint file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(output_path)
    seen = {_observation_key(row) for row in existing if _observation_key(row) is not None}
    session_windows = windows or dict(DEFAULT_SESSION_WINDOWS)
    started = datetime.now(timezone.utc)
    previous_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if resume else {}
    sweep_index = int(previous_checkpoint.get("completed_sweeps", 0))
    end_at = time.monotonic() + duration_minutes * 60
    new_rows: list[dict[str, Any]] = []
    specs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    interrupted = False
    try:
        while time.monotonic() < end_at:
            sweep_index += 1
            sweep_started = datetime.now(timezone.utc)
            for ix, pair in enumerate(pairs, 1):
                progress(f"[{sweep_index}] {ix}/{len(pairs)} {pair}: reading tick", flush=True)
                observation_utc = datetime.now(timezone.utc)
                try:
                    row, spec = _quote_observation(adapter, pair, observation_utc, session_windows)
                    key = _observation_key(row)
                    row["sample_duplicate"] = key is not None and key in seen
                    if key is not None:
                        seen.add(key)
                    specs[pair] = spec
                    new_rows.append(row)
                    with output_path.open("a", encoding="utf-8") as target:
                        target.write(json.dumps(row, allow_nan=False) + "\n")
                        target.flush()
                        os.fsync(target.fileno())
                    progress(f"[{sweep_index}] {ix}/{len(pairs)} {pair}: {row['validation_status']} spread={row['spread_pips']}", flush=True)
                except Exception as exc:
                    error = {"pair": pair, "observation_utc": observation_utc.isoformat(),
                             "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(error)
                    row = {"pair": pair, "observation_utc": observation_utc.isoformat(),
                           "validation_status": "api_error", "validation_error": error["error"],
                           "sample_duplicate": False}
                    new_rows.append(row)
                    with output_path.open("a", encoding="utf-8") as target:
                        target.write(json.dumps(row, allow_nan=False) + "\n")
                        target.flush()
                        os.fsync(target.fileno())
                    progress(f"[{sweep_index}] {ix}/{len(pairs)} {pair}: ERROR {error['error']}", flush=True)
            checkpoint = {"started_at_utc": started.isoformat(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                          "completed_sweeps": sweep_index, "last_sweep_started_utc": sweep_started.isoformat(),
                          "observation_file": str(output_path), "last_completed_pair": pair if 'pair' in locals() else None}
            _atomic_json(checkpoint_path, checkpoint)
            remaining = end_at - time.monotonic()
            if remaining > 0:
                time.sleep(min(interval_seconds, remaining))
    except KeyboardInterrupt:
        interrupted = True
        progress("Stop requested; saved observations and checkpoint are intact.", flush=True)
    return {"started_at_utc": started.isoformat(), "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_minutes_requested": duration_minutes, "interval_seconds": interval_seconds,
            "completed_sweeps": sweep_index - int(previous_checkpoint.get("completed_sweeps", 0)),
            "total_checkpoint_sweeps": sweep_index, "interrupted": interrupted,
            "rows_added": len(new_rows), "errors": errors, "latest_symbol_specifications": list(specs.values()),
            "observation_file": str(output_path), "checkpoint_file": str(checkpoint_path),
            "session_windows_utc": session_windows}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def spread_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame["observation_dt"] = pd.to_datetime(frame.observation_utc, utc=True, errors="coerce")
    if "spread_pips" not in frame:
        frame["spread_pips"] = np.nan
    frame["spread_pips"] = pd.to_numeric(frame["spread_pips"], errors="coerce")
    results = []
    for (pair, session), group in frame.groupby(["pair", "primary_session_utc"], dropna=False, sort=True):
        valid = group[(group.validation_status == "valid") & ~group.sample_duplicate & group.spread_pips.notna()]
        span = ((valid.observation_dt.max() - valid.observation_dt.min()).total_seconds() / 60
                if len(valid) > 1 else 0.0)
        values = valid.spread_pips.astype(float)
        results.append({"pair": pair, "session_utc": session, "raw_observation_count": int(len(group)),
                        "valid_unique_sample_count": int(len(valid)),
                        "invalid_observation_count": int((group.validation_status != "valid").sum()),
                        "duplicate_count": int(group.sample_duplicate.fillna(False).sum()),
                        "observation_coverage_minutes": float(span),
                        "mean_spread_pips": float(values.mean()) if len(values) else None,
                        "median_spread_pips": float(values.median()) if len(values) else None,
                        "min_spread_pips": float(values.min()) if len(values) else None,
                        "max_spread_pips": float(values.max()) if len(values) else None,
                        "std_spread_pips": float(values.std(ddof=1)) if len(values) > 1 else None,
                        "p90_spread_pips": float(values.quantile(.90)) if len(values) else None,
                        "p95_spread_pips": float(values.quantile(.95)) if len(values) else None,
                        "p99_spread_pips": float(values.quantile(.99)) if len(values) else None,
                        "first_observation_utc": group.observation_dt.min().isoformat() if group.observation_dt.notna().any() else None,
                        "last_observation_utc": group.observation_dt.max().isoformat() if group.observation_dt.notna().any() else None,
                        "distribution_status": "sufficient_for_descriptive_demo_spread_summary" if len(valid) >= RELIABLE_SAMPLE_MIN and span >= RELIABLE_COVERAGE_MINUTES else "insufficient_sample_or_coverage"})
    return results


def _r_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "profit_factor": None,
                "total_r": 0.0, "max_drawdown_r": 0.0, "max_consecutive_losses": 0,
                "ci95_iid_low": None, "ci95_iid_high": None}
    ordered = frame.sort_values("entry_timestamp", kind="stable")
    returns = ordered.r_multiple.to_numpy(dtype=float)
    valid = returns[np.isfinite(returns)]
    if not len(valid):
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "profit_factor": None,
                "total_r": 0.0, "max_drawdown_r": 0.0, "max_consecutive_losses": 0,
                "ci95_iid_low": None, "ci95_iid_high": None}
    gains = float(valid[valid > 0].sum())
    losses = float(-valid[valid < 0].sum())
    equity = np.r_[0.0, np.cumsum(valid)]
    dd = np.maximum.accumulate(equity) - equity
    streak = maximum_streak = 0
    for value in valid:
        streak = streak + 1 if value < 0 else 0
        maximum_streak = max(maximum_streak, streak)
    mean = float(valid.mean())
    sem = float(valid.std(ddof=1) / math.sqrt(len(valid))) if len(valid) > 1 else None
    return {"trades": int(len(valid)), "win_rate": float((valid > 0).mean()),
            "expectancy_r": mean,
            "profit_factor": gains / losses if losses else ("infinite" if gains else None),
            "total_r": float(valid.sum()), "max_drawdown_r": float(dd.max()),
            "max_consecutive_losses": int(maximum_streak),
            "ci95_iid_low": mean - 1.96 * sem if sem is not None else None,
            "ci95_iid_high": mean + 1.96 * sem if sem is not None else None}


def _group_summaries(trades: pd.DataFrame, dimensions: list[str]) -> list[dict[str, Any]]:
    keys = ["split"] + dimensions
    output = []
    for group_key, group in trades.groupby(keys, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        output.append({**dict(zip(keys, group_key)), **_r_summary(group)})
    return output


def _duration_summaries(trades: pd.DataFrame) -> list[dict[str, Any]]:
    work = trades.copy()
    work["duration_minutes"] = (work.exit_timestamp - work.entry_timestamp).dt.total_seconds() / 60
    output = []
    keys = ["strategy_id", "pair", "timeframe", "split"]
    for key, group in work.groupby(keys, sort=False):
        values = group.duration_minutes.astype(float)
        output.append({**dict(zip(keys, key)), "trades": len(group),
                       "mean_duration_minutes": float(values.mean()),
                       "median_duration_minutes": float(values.median()),
                       "p90_duration_minutes": float(values.quantile(.9)),
                       "max_duration_minutes": float(values.max())})
    return output


def _exit_outcomes(trades: pd.DataFrame) -> list[dict[str, Any]]:
    keys = ["strategy_id", "pair", "timeframe", "split", "exit_reason"]
    denominators = trades.groupby(keys[:-1], sort=False).size()
    return [{**dict(zip(keys, key)), "trades": int(len(group)),
             "share_of_strategy_cell": float(len(group) / denominators.get(key[:-1], 1))}
            for key, group in trades.groupby(keys, sort=False)]


def _hhhl_diagnostic(trades: pd.DataFrame, mission9: dict[str, Any]) -> dict[str, Any]:
    frame = trades[(trades.strategy_id == "structure_hh_hl") & (trades.split == "validation") &
                   (trades.regime == "high_volatility")].copy()
    frame["r_multiple"] = frame.gross_pnl_pips / frame.stop_distance_pips
    by_combo = []
    rng = np.random.default_rng(20261110)
    for (pair, timeframe), group in frame.groupby(["pair", "timeframe"], sort=False):
        ordered = group.sort_values("entry_timestamp")
        values = ordered.r_multiple.to_numpy(float)
        n = len(values)
        if n >= 10:
            block = min(5, n)
            samples = np.empty(1000, dtype=float)
            blocks_needed = int(math.ceil(n / block))
            starts = rng.integers(0, n, size=(1000, blocks_needed))
            for i in range(1000):
                sampled = np.concatenate([values[(start + np.arange(block)) % n] for start in starts[i]])[:n]
                samples[i] = sampled.mean()
            ci_low, ci_high = np.quantile(samples, [.025, .975]).tolist()
        else:
            ci_low = ci_high = None
        wins = values[values > 0]
        sorted_wins = np.sort(wins)[::-1]
        cost = (mission9.get("market_regime") or [])
        stress_rows = [x for x in cost if x["strategy_id"] == "structure_hh_hl" and x["pair"] == pair
                       and x["timeframe"] == timeframe and x["split"] == "validation"
                       and x["regime"] == "high_volatility" and x["scenario"] == MISSION9_STRESS_SCENARIO]
        gross_rows = [x for x in cost if x["strategy_id"] == "structure_hh_hl" and x["pair"] == pair
                      and x["timeframe"] == timeframe and x["split"] == "validation"
                      and x["regime"] == "high_volatility" and x["scenario"] == "spread_0_commission_0_slippage_0_pips_rt"]
        by_combo.append({"pair": pair, "timeframe": timeframe, "trades": n,
                         "gross_expectancy_r": float(values.mean()) if n else None,
                         "moving_block_bootstrap_95ci_low": float(ci_low) if ci_low is not None else None,
                         "moving_block_bootstrap_95ci_high": float(ci_high) if ci_high is not None else None,
                         "iid_ci95_low": float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                         "positive_gross_r_sum": float(wins.sum()),
                         "top_one_winner_share_of_positive_r": float(sorted_wins[:1].sum() / wins.sum()) if wins.sum() else None,
                         "top_five_winners_share_of_positive_r": float(sorted_wins[:5].sum() / wins.sum()) if wins.sum() else None,
                         "expectancy_excluding_top_five_winners_r": float((values.sum() - sorted_wins[:5].sum()) / n) if n else None,
                         "gross_mission9_metrics": gross_rows[0] if gross_rows else None,
                         "highest_cost_sensitivity_metrics": stress_rows[0] if stress_rows else None})
    return {"strategy_id": "structure_hh_hl", "split": "validation only", "regime": "high_volatility",
            "trade_count": int(len(frame)), "pair_timeframe_combinations": len(by_combo),
            "positive_pair_timeframe_combinations": sum(x["gross_expectancy_r"] is not None and x["gross_expectancy_r"] > 0 for x in by_combo),
            "median_cell_expectancy_r": float(np.median([x["gross_expectancy_r"] for x in by_combo if x["gross_expectancy_r"] is not None])) if by_combo else None,
            "combination_diagnostics": by_combo,
            "bootstrap_method": "circular moving-block bootstrap, block length 5 trades, 1,000 resamples per pair/timeframe; descriptive only; does not establish independence or profitability"}


def _equity_curve_audit(path: Path, trades: pd.DataFrame) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    counts: dict[tuple[str, str, str], int] = {}
    max_dd: dict[tuple[str, str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        for row in reader:
            key = (row["strategy_id"], row["pair"], row["timeframe"])
            counts[key] = counts.get(key, 0) + 1
            dd = float(row["drawdown_gross_pips"])
            max_dd[key] = max(max_dd.get(key, 0.0), dd)
    trade_counts = trades.groupby(["strategy_id", "pair", "timeframe"], sort=False).size()
    mismatches = []
    for key, count in counts.items():
        if int(trade_counts.get(key, 0)) != count:
            mismatches.append({"strategy_id": key[0], "pair": key[1], "timeframe": key[2],
                               "curve_rows": count, "trade_rows": int(trade_counts.get(key, 0))})
    return {"path": str(path), "columns": columns, "curve_rows": sum(counts.values()),
            "strategy_cells": len(counts), "trade_curve_row_count_mismatches": mismatches,
            "max_drawdown_gross_pips_by_cell": [{"strategy_id": k[0], "pair": k[1], "timeframe": k[2], "max_drawdown_gross_pips": v} for k, v in max_dd.items()]}


def diagnose_saved_results(mission8_path: Path, mission9_path: Path, trade_path: Path,
                           equity_path: Path) -> dict[str, Any]:
    mission8 = json.loads(mission8_path.read_text(encoding="utf-8"))
    mission9 = json.loads(mission9_path.read_text(encoding="utf-8"))
    cols = ["strategy_id", "pair", "timeframe", "decision_timestamp", "entry_timestamp", "exit_timestamp",
            "direction", "gross_pnl_pips", "stop_distance_pips", "risk_reward", "exit_reason", "regime",
            "session_utc", "position_units", "risk_amount", "gross_pnl_account_currency", "signal_evidence"]
    trades = pd.read_csv(trade_path, usecols=cols, low_memory=False)
    for col in ("decision_timestamp", "entry_timestamp", "exit_timestamp"):
        trades[col] = pd.to_datetime(trades[col], utc=True, errors="raise")
    for col in ("gross_pnl_pips", "stop_distance_pips", "risk_reward", "position_units", "risk_amount", "gross_pnl_account_currency"):
        trades[col] = pd.to_numeric(trades[col], errors="coerce")
    raw_trade_rows = len(trades)
    raw_trade_cells = trades[["strategy_id", "pair", "timeframe"]].copy()
    bounds = _research_split_bounds(mission8) if mission8.get("data_exclusions_and_provenance") else None
    trades = _assign_splits(trades, mission8["results"], bounds)
    purged = int((trades.split == "purged_boundary_crossing_or_outside").sum())
    trades = trades[trades.split != "purged_boundary_crossing_or_outside"].copy()
    trades["r_multiple"] = trades.gross_pnl_pips / trades.stop_distance_pips
    trades["duration_minutes"] = (trades.exit_timestamp - trades.entry_timestamp).dt.total_seconds() / 60
    trades["holding_bars_estimate"] = trades.duration_minutes / trades.timeframe.map({"M15": 15, "H1": 60, "H4": 240})
    group_dims = {
        "by_pair": ["pair"], "by_timeframe": ["timeframe"], "by_strategy": ["strategy_id"],
        "by_market_regime": ["regime"], "by_session_utc": ["session_utc"],
        "long_vs_short": ["direction"],
        "strategy_pair_timeframe": ["strategy_id", "pair", "timeframe"],
        "regime_pair_timeframe": ["regime", "pair", "timeframe"],
        "session_pair_timeframe": ["session_utc", "pair", "timeframe"],
    }
    summaries = {name: _group_summaries(trades, dims) for name, dims in group_dims.items()}
    exits = _exit_outcomes(trades)
    duration = _duration_summaries(trades)
    relationships = mission9.get("complementarity_analysis_validation_only", [])
    # Equity curves are built from the full raw trade ledger, including rows
    # later excluded from chronological split diagnostics. Compare like with like.
    equity_audit = _equity_curve_audit(equity_path, raw_trade_cells)
    hhhl = _hhhl_diagnostic(trades, mission9)
    scenario_rows = [x for x in mission9.get("market_regime", []) if x["strategy_id"] == "structure_hh_hl"
                     and x["regime"] == "high_volatility" and x["split"] == "validation"]
    hhhl_cost_sensitivity = []
    if scenario_rows:
        frame = pd.DataFrame(scenario_rows)
        for scenario, group in frame.groupby("scenario", sort=False):
            hhhl_cost_sensitivity.append({"scenario": scenario, "pair_timeframe_observations": int(len(group)),
                                          "trades": int(group.trades.sum()),
                                          "median_expectancy_r": float(group.expectancy_r.median()),
                                          "positive_observations": int((group.expectancy_r > 0).sum()),
                                          "maximum_cost_components_are_assumptions": scenario != "spread_0_commission_0_slippage_0_pips_rt"})
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": {"mission8_json": str(mission8_path), "mission9_json": str(mission9_path),
                             "trade_log": str(trade_path), "equity_curve": str(equity_path)},
            "mission8_source_generated_at_utc": mission8.get("generated_at_utc"),
            "mission9_analyzed_at_utc": mission9.get("analyzed_at_utc"),
            "raw_trade_rows": int(raw_trade_rows), "in_period_trade_rows": int(len(trades)),
            "boundary_crossing_or_outside_rows_excluded": purged,
            "split_policy": "saved Mission 8 chronological 60/20/20 boundaries from existing research Parquet; validation informs hypothesis review; test remains descriptive holdout only",
            "gross_diagnostics": summaries, "duration_and_holding": duration,
            "stop_target_and_exit_outcomes": exits,
            "high_volatility_structure_hh_hl": hhhl,
            "structure_hh_hl_high_volatility_cost_sensitivity": hhhl_cost_sensitivity,
            "existing_strategy_relationships": relationships,
            "existing_mission9_cost_scenarios": mission9.get("cost_scenarios", []),
            "existing_mission9_candidate_assessment": mission9.get("candidate_assessment_validation_only", []),
            "existing_equity_curve_audit": equity_audit,
            "uncertainty_method": "IID normal confidence intervals are provided only as rough diagnostics. They do not correct for serial dependence, cross-pair dependence, or selection bias; HHHL also receives a moving-block bootstrap by pair/timeframe.",
            "test_period_used_for_optimization_or_candidate_labels": False}


def _account_cost_summary(adapter: MT5DataAdapter) -> dict[str, Any]:
    account = adapter.account_info()
    if account is None:
        account_fields = {}
    else:
        account_fields = {name: getattr(account, name, None) for name in
                          ("company", "server", "currency", "trade_mode", "margin_mode", "trade_allowed")}
    account_fields["commission_blocked"] = getattr(account, "commission_blocked", None) if account else None
    account_fields["commission_blocked_is_not_a_rate"] = True
    now = datetime.now(timezone.utc)
    deal_summary: dict[str, Any]
    try:
        deals = adapter.deal_history(now - timedelta(days=90), now)
        nonzero = [x for x in deals if float(getattr(x, "commission", 0) or 0) != 0]
        deal_summary = {"window_days": 90, "deals_returned": len(deals),
                        "deals_with_nonzero_commission": len(nonzero),
                        "nonzero_commission_sum_account_currency": sum(float(getattr(x, "commission", 0) or 0) for x in nonzero),
                        "commission_rate_established": False,
                        "limitation": "No nonzero commission charges in the bounded retrieved history; this cannot establish a zero-commission schedule."}
    except Exception as exc:
        deal_summary = {"window_days": 90, "error": f"{type(exc).__name__}: {exc}", "commission_rate_established": False}
    return {"account_metadata": account_fields, "deal_history_summary": deal_summary,
            "commission_schedule": {"status": "unavailable", "per_lot": None, "per_side_or_round_trip": None,
                                    "minimum_fee": None, "account_conditions": None}}


def _distribution_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_reports(report: dict[str, Any], output_dir: Path, *, stamp: str | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"mission10_broker_calibration_{stamp}"
    paths = {"json": base.with_suffix(".json"), "markdown": base.with_suffix(".md"),
             "spread_csv": output_dir / f"mission10_spread_distributions_{stamp}.csv",
             "specs_csv": output_dir / f"mission10_symbol_specs_{stamp}.csv",
             "diagnostic_csv": output_dir / f"mission10_strategy_diagnostics_{stamp}.csv"}
    report["report_files"] = {key: str(value) for key, value in paths.items()}
    _atomic_json(paths["json"], report)
    _distribution_rows_csv(report.get("spread_distributions", []), paths["spread_csv"])
    _distribution_rows_csv(report.get("symbol_specifications", []), paths["specs_csv"])
    _distribution_rows_csv(report.get("strategy_diagnostic_flattened", []), paths["diagnostic_csv"])
    live = report.get("spread_collection", {})
    lines = ["# Mission 10 Broker Cost Calibration and Strategy Diagnostics", "",
             f"Generated UTC: {report['generated_at_utc']}", "",
             "## Timestamp investigation", "",
             report.get("timestamp_root_cause", ""), "",
             f"Live quote verification: {report.get('live_timestamp_verification', {}).get('valid_pairs', 0)}/{report.get('live_timestamp_verification', {}).get('total_pairs', 0)} pairs had fresh, valid normalized UTC timestamps.",
             f"Quote/candle alignment: {report.get('live_timestamp_verification', {}).get('candle_alignment_summary', 'not measured')}", "",
             "Raw integer tick epochs are retained separately from normalized UTC fields. Future/stale checks remain enabled.", "",
             "## Spread observations", "",
             f"Collection added {live.get('rows_added', 0)} rows in {live.get('completed_sweeps', 0)} completed sweep(s); files append and checkpoint for safe resumption.",
             f"The descriptive distribution threshold is {RELIABLE_SAMPLE_MIN} unique valid samples and {RELIABLE_COVERAGE_MINUTES} minutes per pair/session.", "",
             "| Pair | Session UTC | Valid unique samples | Raw samples | Duplicates | Median pips | P95 pips | Status |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in report.get("spread_distributions", []):
        median = "n/a" if row["median_spread_pips"] is None else f"{row['median_spread_pips']:.4f}"
        p95 = "n/a" if row["p95_spread_pips"] is None else f"{row['p95_spread_pips']:.4f}"
        lines.append(f"| {row['pair']} | {row['session_utc']} | {row['valid_unique_sample_count']} | {row['raw_observation_count']} | {row['duplicate_count']} | {median} | {p95} | {row['distribution_status']} |")
    commission = report.get("commission", {})
    lines.extend(["", "## Commission and contract specifications", "",
                  "Commission schedule: unavailable unless explicitly marked otherwise in the JSON. No rate is inferred from commission_blocked or from absent deal charges.",
                  f"Recent deal history: {commission.get('deal_history_summary', {}).get('deals_returned', 'unavailable')} deals in the bounded lookback; nonzero commission deals: {commission.get('deal_history_summary', {}).get('deals_with_nonzero_commission', 'unavailable')}.",
                  "Symbol metadata CSV includes contract size, tick size/value, point, digits, volume bounds/step, execution/trading mode, and swaps.", "",
                  "## Strategy diagnostic summary", "",
                  f"Mission 8 trade rows read: {report.get('strategy_diagnostics', {}).get('raw_trade_rows', 'n/a')}; in-period rows evaluated: {report.get('strategy_diagnostics', {}).get('in_period_trade_rows', 'n/a')}; boundary/outside rows excluded: {report.get('strategy_diagnostics', {}).get('boundary_crossing_or_outside_rows_excluded', 'n/a')}.",
                  "All Mission 9 metrics and trade-level diagnostics are descriptive. The Mission 8 test split was not used to optimize or label candidates.", "",
                  "## `structure_hh_hl` high-volatility investigation", "",
                  f"Validation high-volatility trades: {report.get('strategy_diagnostics', {}).get('high_volatility_structure_hh_hl', {}).get('trade_count', 0)} across {report.get('strategy_diagnostics', {}).get('high_volatility_structure_hh_hl', {}).get('pair_timeframe_combinations', 0)} pair/timeframe combinations.",
                  "Per-pair/timeframe bootstrap intervals, cost sensitivities, trade concentration, and exits are in JSON.", "",
                  "## Limits", "",
                  "Observed MetaQuotes-Demo costs are not a forecast for a future live broker. Current snapshot spreads are not applied retrospectively to every historical trade.",
                  "Spread distributions below the stated sample/coverage threshold are not reliable estimates. Financing and commission remain separate from measured spread.",
                  "No orders, model training, model promotion, or original dataset changes were performed."])
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def _latest(directory: Path, pattern: str) -> Path:
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files match {pattern} under {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-spreads", action="store_true", help="Collect bounded read-only MT5 observations")
    parser.add_argument("--duration-minutes", type=float, default=2.0)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true", help="Resume by appending to existing observations and checkpoint")
    parser.add_argument("--sessions", action="append", default=None, metavar="NAME=START-END",
                        help="Override UTC session window, e.g. --sessions London=07:00-13:00")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--history-file", type=Path, default=ROOT / "reports/strategy-research/mission10/spread_observations.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "reports/strategy-research/mission10/spread_checkpoint.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/strategy-research/mission10")
    parser.add_argument("--mission8-json", type=Path)
    parser.add_argument("--mission9-json", type=Path)
    parser.add_argument("--trade-log", type=Path)
    parser.add_argument("--equity-curve", type=Path)
    parser.add_argument("--terminal-path", default=None)
    parser.add_argument("--server-timezone", default=None)
    args = parser.parse_args(argv)
    reports_dir = ROOT / "reports/strategy-research"
    mission8_path = args.mission8_json or _latest(reports_dir, "strategy_research_*.json")
    mission9_path = args.mission9_json or _latest(reports_dir / "mission9", "mission9_cost_research_*.json")
    trade_path = args.trade_log or _latest(reports_dir, "trade_log_*.csv")
    equity_path = args.equity_curve or _latest(reports_dir, "equity_drawdown_*.csv")
    session_windows = parse_session_windows(args.sessions)
    cfg = AppConfig.from_env()
    timezone_name = args.server_timezone or cfg.mt5_server_timezone
    adapter = MT5DataAdapter(terminal_path=args.terminal_path or cfg.mt5_terminal_path,
                             server_timezone=timezone_name,
                             max_quote_age_seconds=cfg.max_quote_age_seconds)
    now = datetime.now(timezone.utc)
    verification: dict[str, Any] = {"valid_pairs": 0, "total_pairs": 0, "pairs": [],
                                    "candle_alignment_summary": "not measured"}
    collection: dict[str, Any] = {"rows_added": 0, "completed_sweeps": 0, "errors": []}
    specs: list[dict[str, Any]] = []
    commission: dict[str, Any] = {}
    if args.collect_spreads:
        try:
            adapter.connect()
            specs = [symbol_specification(adapter, pair) for pair in args.pairs.split(",") if pair]
            commission = _account_cost_summary(adapter)
            # Verify each pair and one M5 candle clock on the same normalized UTC axis.
            alignments = []
            for pair in args.pairs.split(","):
                if not pair:
                    continue
                try:
                    broker = adapter.resolve_symbol(pair)
                    quote = adapter.quote(broker)
                    bars = adapter.candles(broker, "M5", count=5)
                    last_bar = bars.time.iloc[-1].to_pydatetime()
                    delta = (quote.time - last_bar).total_seconds()
                    alignments.append({"pair": pair, "broker_symbol": broker, "raw_tick_seconds": quote.raw_time_seconds,
                                       "raw_tick_milliseconds": quote.raw_time_milliseconds,
                                       "normalized_tick_utc": quote.time.isoformat(),
                                       "quote_age_seconds": (datetime.now(timezone.utc) - quote.time).total_seconds(),
                                       "latest_completed_m5_candle_utc": last_bar.isoformat(),
                                       "tick_minus_candle_seconds": delta, "validation_status": "valid"})
                except Exception as exc:
                    alignments.append({"pair": pair, "validation_status": "error",
                                       "error": f"{type(exc).__name__}: {exc}"})
            verification = {"valid_pairs": sum(x["validation_status"] == "valid" for x in alignments),
                            "total_pairs": len(alignments), "pairs": alignments,
                            "candle_alignment_summary": "Each available M5 bar closed approximately five minutes before its normalized live tick; see per-pair seconds in JSON."}
            collection = collect_spreads(adapter, args.pairs.split(","), duration_minutes=args.duration_minutes,
                                         interval_seconds=args.interval_seconds,
                                         output_path=args.history_file, checkpoint_path=args.checkpoint,
                                         resume=args.resume, windows=session_windows)
            specs = [symbol_specification(adapter, pair) for pair in args.pairs.split(",") if pair]
        except Exception as exc:
            collection.setdefault("errors", []).append({"pair": "*", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            adapter.close()
    history_rows = load_jsonl(args.history_file)
    seen: set[tuple[Any, ...]] = set()
    for row in history_rows:
        key = _observation_key(row)
        if key is not None and key in seen:
            row["sample_duplicate"] = True
        if key is not None:
            seen.add(key)
    distributions = spread_statistics(history_rows)
    diagnostics = diagnose_saved_results(mission8_path, mission9_path, trade_path, equity_path)
    flat_diagnostics = diagnostics["gross_diagnostics"]["strategy_pair_timeframe"]
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "timezone_configuration": timezone_name,
              "timestamp_root_cause": (
                  "Mission 9 spread collection bypassed MT5DataAdapter and interpreted raw tick.time/time_msc directly as UTC with datetime.fromtimestamp(..., UTC). On the connected MetaQuotes-Demo terminal, all seven raw tick epochs were exactly about two hours ahead when read as UTC. The existing configured server-wall conversion with Etc/GMT-2 normalized each tick to within about two seconds of the host UTC clock; latest completed M5 candles normalized to about five minutes before the same ticks. The runtime environment had no MT5_SERVER_TIMEZONE override, so AppConfig also fell back to UTC. The collector then independently bypassed the adapter conversion, so setting the variable alone would not fix Mission 9 observations."),
              "timestamp_fix": "The adapter now exposes raw tick fields separately, uses one normalization function for tick and candle paths, and validates freshness/future times after normalization. AppConfig defaults to the live-verified MetaQuotes-Demo Etc/GMT-2 offset; MT5_SERVER_TIMEZONE remains an override and must be rechecked around broker offset/DST changes.",
              "live_timestamp_verification": verification,
              "spread_collection": collection,
              "spread_history_file": str(args.history_file),
              "spread_distributions": distributions,
              "session_definitions_utc": session_windows,
              "session_coverage": {name: any(name in row.get("session_tags_utc", []) for row in history_rows) for name in SESSION_NAMES},
              "sessions_covered": [name for name in SESSION_NAMES if any(name in row.get("session_tags_utc", []) for row in history_rows)],
              "all_requested_sessions_covered": all(any(name in row.get("session_tags_utc", []) for row in history_rows) for name in SESSION_NAMES),
              "symbol_specifications": specs,
              "commission": commission,
              "strategy_diagnostics": diagnostics,
              "strategy_diagnostic_flattened": flat_diagnostics,
              "measured_vs_assumed_costs": {"current_spreads": "measured MetaQuotes-Demo observations; sample/coverage adequacy is explicit per pair/session; not retroactively applied to historical trades",
                                            "commission": "unknown unless verified from broker schedule; available symbol fields and bounded deal history inspected",
                                            "slippage": "assumption only",
                                            "swap": "long/short swap values read from symbol metadata; financing calculations are not modeled in the saved fixed-duration trades"},
              "safety": {"read_only": True, "orders_submitted": False, "account_modified": False,
                         "market_data_modified": False, "model_training_or_promotion": False,
                         "test_set_used_for_optimization": False}}
    paths = write_reports(report, args.output_dir)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(f"Live quotes validated: {verification['valid_pairs']}/{verification['total_pairs']}; "
          f"spread rows added: {collection.get('rows_added', 0)}; "
          f"sessions covered: {', '.join(report['sessions_covered']) or 'none'}")
    return 0 if not collection.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
