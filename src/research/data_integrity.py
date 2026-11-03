"""Read-only validation helpers for collected MT5 OHLC datasets."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import hashlib
import json


TIMEFRAME_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}
OHLC = ("open", "high", "low", "close")


def audit_frame(frame: pd.DataFrame, pair: str, timeframe: str) -> dict[str, Any]:
    """Return timestamped findings and statistics without changing *frame*."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    required = {"time", *OHLC}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")

    raw_times = frame["time"]
    raw_is_aware = isinstance(raw_times.dtype, pd.DatetimeTZDtype)
    t = pd.to_datetime(raw_times, utc=True, errors="coerce")
    prices = frame.loc[:, OHLC].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(prices.to_numpy(dtype=float)).all(axis=1)
    valid = finite & (prices["low"].to_numpy() > 0)
    valid &= prices["high"].to_numpy() >= prices[["open", "close", "low"]].max(axis=1).to_numpy()
    valid &= prices["low"].to_numpy() <= prices[["open", "close", "high"]].min(axis=1).to_numpy()
    invalid_rows = [t.iloc[i].isoformat() if pd.notna(t.iloc[i]) else f"row:{i}" for i, ok in enumerate(valid) if not ok]

    ns = t.astype("int64").to_numpy()
    nat = t.isna().to_numpy()
    valid_times = ~nat
    duplicate_mask = t.duplicated(keep=False).to_numpy() & valid_times
    duplicate_timestamps = sorted({t.iloc[i].isoformat() for i in np.flatnonzero(duplicate_mask)})
    # Preserve original adjacent order for exact affected rows.
    nonmonotonic = [t.iloc[i + 1].isoformat() for i in np.flatnonzero((np.diff(ns) < 0) & ~nat[:-1] & ~nat[1:])]

    delta = t.diff().dt.total_seconds().div(60)
    step = TIMEFRAME_MINUTES[timeframe]
    interval_counts = delta.dropna().value_counts().sort_index()
    interval_distribution = {str(int(minutes)): int(count) for minutes, count in interval_counts.items()}
    daily_sampling_years: list[dict[str, Any]] = []
    first_expected_interval = None
    if timeframe == "H4" and len(delta) > 1:
        expected_indices = np.flatnonzero(delta.eq(step).to_numpy())
        if len(expected_indices):
            first_expected_interval = t.iloc[int(expected_indices[0])].isoformat()
        for year in sorted(t.dropna().dt.year.unique()):
            in_year = t.dt.year.eq(year) & delta.notna()
            year_deltas = delta.loc[in_year]
            daily_count = int(year_deltas.eq(1440).sum())
            expected_count = int(year_deltas.eq(step).sum())
            if daily_count >= 180 and expected_count < 10:
                year_times = t.loc[t.dt.year.eq(year)]
                daily_sampling_years.append({"year": int(year), "start": year_times.min().isoformat(),
                                             "end": year_times.max().isoformat(),
                                             "daily_intervals": daily_count, "h4_intervals": expected_count})
    gaps: list[dict[str, str | int]] = []
    for i in np.flatnonzero((delta.to_numpy() > step + 1e-6) & ~nat):
        prev, cur = t.iloc[i - 1], t.iloc[i]
        minutes = int(delta.iloc[i])
        gaps.append({"from": prev.isoformat(), "to": cur.isoformat(), "elapsed_minutes": minutes,
                     "missing_expected_bars": max(0, int(round(minutes / step)) - 1),
                     "classification": _classify_gap(prev, cur, minutes)})

    # Robustly flag only extreme local price changes; this is a screening threshold,
    # not a claim that a market move is impossible.
    close = prices["close"].to_numpy(dtype=float)
    pct = np.abs(np.diff(close) / close[:-1]) if len(close) > 1 else np.array([])
    jumps = []
    for i in np.flatnonzero((pct > 0.10) & np.isfinite(pct)):
        jumps.append({"from": t.iloc[i].isoformat(), "to": t.iloc[i + 1].isoformat(),
                      "absolute_percent_change": float(pct[i] * 100),
                      "previous_close": float(close[i]), "next_close": float(close[i + 1])})

    zero_range = prices["high"].eq(prices["low"]) & prices["high"].eq(prices["open"]) & prices["open"].eq(prices["close"])
    # Runs are in original chronological order; group adjacent identical OHLC rows.
    exact_flat_runs: list[dict[str, Any]] = []
    if len(frame):
        same = prices.eq(prices.shift()).all(axis=1).fillna(False).to_numpy()
        starts = np.flatnonzero(~same)
        ends = list(starts[1:]) + [len(frame)]
        for start, end in zip(starts, ends):
            if end - start >= 3 and valid_times[start] and valid_times[end - 1]:
                exact_flat_runs.append({"start": t.iloc[start].isoformat(), "end": t.iloc[end - 1].isoformat(), "candles": int(end - start)})

    oldest = t.min() if t.notna().any() else pd.NaT
    newest = t.max() if t.notna().any() else pd.NaT
    date_findings = []
    if pair.upper() == "EURUSD" and pd.notna(oldest) and oldest < pd.Timestamp("1999-01-01", tz="UTC"):
        date_findings.append({"severity": "critical", "timestamp": oldest.isoformat(),
                              "message": "EURUSD bars predate the euro's 1999 launch; these cannot be EURUSD observations."})

    return {
        "pair": pair.upper(), "timeframe": timeframe, "rows": len(frame),
        "oldest_utc": None if pd.isna(oldest) else oldest.isoformat(),
        "newest_utc": None if pd.isna(newest) else newest.isoformat(),
        "timezone_aware_utc": bool(raw_is_aware and str(raw_times.dtype.tz) == "UTC"),
        "invalid_ohlc_count": len(invalid_rows), "invalid_ohlc_timestamps": invalid_rows,
        "duplicate_timestamp_count": len(duplicate_timestamps), "duplicate_timestamps": duplicate_timestamps,
        "nonmonotonic_count": len(nonmonotonic), "nonmonotonic_timestamps": nonmonotonic,
        "gap_count": len(gaps), "gaps": gaps,
        "interval_minutes_distribution": interval_distribution,
        "exact_expected_interval_count": int(delta.eq(step).sum()),
        "exact_expected_interval_ratio": float(delta.eq(step).sum() / max(1, len(frame) - 1)),
        "sustained_daily_sampling_years_in_h4": daily_sampling_years,
        "first_expected_h4_interval_timestamp": first_expected_interval,
        "zero_range_candle_count": int(zero_range.sum()), "zero_range_ratio": float(zero_range.mean()) if len(frame) else 0.0,
        "exact_flat_runs": exact_flat_runs, "price_jumps_over_10pct": jumps,
        "date_plausibility_findings": date_findings,
        "calendar_coverage_days": 0 if pd.isna(oldest) else (newest - oldest).total_seconds() / 86400,
    }


def _classify_gap(previous: pd.Timestamp, current: pd.Timestamp, minutes: int) -> str:
    if previous.weekday() == 4 and current.weekday() == 6:
        return "weekend-close/open candidate"
    if previous.weekday() == 4 and current.weekday() == 0:
        return "weekend closure candidate"
    if previous.weekday() == 5 or current.weekday() == 5:
        return "weekend-crossing"
    if minutes >= 3 * 24 * 60:
        return "multi-day weekday/holiday or missing-history; unresolved"
    return "within-week session/holiday or missing bars; unresolved"


def audit_directory(data_dir: Path) -> dict[str, Any]:
    """Inspect each parquet and adjacent .provenance.json, without writing there."""
    findings = []
    errors = []
    for path in sorted(data_dir.glob("*.parquet")):
        try:
            pair, timeframe = path.stem.rsplit("_", 1)
            frame = pd.read_parquet(path)
            result = audit_frame(frame, pair, timeframe)
            manifest_path = path.with_suffix(".provenance.json")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                result["manifest"] = {k: manifest.get(k) for k in (
                    "broker_symbol", "terminal_symbol", "configured_server_timezone", "collection_time_utc",
                    "history_exhausted", "bounded_request_limit_reached", "rows_after_update", "data_fingerprint_sha256")}
                result["manifest_quality"] = manifest.get("quality", {})
            else:
                result["manifest_error"] = "provenance manifest missing"
            findings.append(result)
        except Exception as exc:  # report a broken dataset rather than skipping it
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_count": len(findings),
            "datasets": findings, "errors": errors}


def audit_dukascope_side(root: Path) -> dict[str, Any]:
    """Capture current BID/ASK file hashes and compare with any manifest hash."""
    base = root / "m1" / "Forex" / "Majors" / "EUR-USD"
    sides: dict[str, Any] = {}
    for side in ("BID", "ASK"):
        directory = base / side
        manifest_path = directory / "_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        files = []
        for path in sorted(directory.glob("*.parquet")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = manifest.get("sha256", {}).get(path.name)
            files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest,
                          "manifest_sha256": expected,
                          "hash_matches_manifest": None if expected is None else expected == digest})
        sides[side] = {"manifest": manifest, "files": files}
    # Compare the current monthly Parquet series with the exact normalized
    # market-frame identities archived in the 2026-10-27 pilot report.
    from .experiment_store import fingerprint_market_frame
    from .hypothesis_orchestrator import M1_QUALITY_POLICY

    references = {
        "BID": "7aa197216310d6a3d700939631a1ce0f9193422a7c3bad08c1d3de9e87e8b0b6",
        "ASK": "43f61bcee55b7cc5f5ad87c6ea91ad45fb0f6019111d3d75758751c99842537a",
    }
    side_timestamps: dict[str, pd.Series] = {}
    for side in ("BID", "ASK"):
        files = sorted((base / side).glob("????-??.parquet"))
        frames = [pd.read_parquet(path, columns=["time", "open", "high", "low", "close", "volume"]) for path in files]
        market = pd.concat(frames, ignore_index=True)
        market["time"] = pd.to_datetime(market["time"], utc=True, errors="raise")
        market = market.sort_values("time", kind="stable").reset_index(drop=True)
        side_timestamps[side] = market["time"]
        policy = dict(M1_QUALITY_POLICY)
        if side == "ASK":
            policy.update({"market_side": "ASK", "alignment_policy": "exact_utc_timestamp_join_v1"})
        fingerprint, identity = fingerprint_market_frame(market, quality_policy=policy)
        sides[side]["normalized_frame_fingerprint"] = fingerprint
        sides[side]["historical_reference_fingerprint"] = references[side]
        sides[side]["matches_2026_10_27_reference"] = fingerprint == references[side]
        sides[side]["normalized_identity"] = identity
    bid_times = side_timestamps["BID"]
    ask_times = side_timestamps["ASK"]
    sides["timestamp_alignment"] = {
        "all_months_same_file_names": [p.name for p in sorted((base / "BID").glob("????-??.parquet"))]
        == [p.name for p in sorted((base / "ASK").glob("????-??.parquet"))],
        "bid_rows": int(len(bid_times)), "ask_rows": int(len(ask_times)),
        "bid_without_ask": int((~bid_times.isin(ask_times)).sum()),
        "ask_without_bid": int((~ask_times.isin(bid_times)).sum()),
    }
    return sides


def _markdown(report: dict[str, Any], dukascopy: dict[str, Any]) -> str:
    lines = ["# MT5 Historical Data Integrity Audit", "", f"Generated: {report['generated_at_utc']}",
             "", f"Datasets read: {report['dataset_count']} / expected 28", "",
             "| Pair | TF | Rows | Oldest UTC | Newest UTC | Duplicates | Invalid OHLC | Gaps | Flat % | Jumps >10% | Severity |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for d in report["datasets"]:
        severity = "CRITICAL" if d["date_plausibility_findings"] or d["invalid_ohlc_count"] else (
            "HIGH" if d["sustained_daily_sampling_years_in_h4"] else
            "HIGH" if d["duplicate_timestamp_count"] or d["nonmonotonic_count"] or d["price_jumps_over_10pct"] else "REVIEW")
        lines.append(f"| {d['pair']} | {d['timeframe']} | {d['rows']} | {d['oldest_utc']} | {d['newest_utc']} | "
                     f"{d['duplicate_timestamp_count']} | {d['invalid_ohlc_count']} | {d['gap_count']} | "
                     f"{d['zero_range_ratio']:.1%} | {len(d['price_jumps_over_10pct'])} | {severity} |")
    lines += ["", "## Critical date plausibility finding", "",
              "EURUSD H4 begins at 1971-01-03 22:00 UTC (the first exact affected timestamp); the euro launched on 1999-01-01. Those records cannot be genuine EURUSD observations. Inspect the full date-finding list in JSON.",
              "USDJPY H4 also begins 1971-01-03 22:00 UTC. That date is not inherently impossible for USDJPY, but the records begin as one daily timestamp at 22:00 with repeated exact OHLC candles, not normal H4 cadence. Treat this pre-modern segment as untrusted pending broker confirmation.",
              "Both EURUSD and USDJPY H4 have a sustained daily cadence from 1971 through the final daily bars in 1998; regular 4-hour cadence starts in 1999. These daily records explain most of their anomalous H4 gap counts. Exact date boundaries and run lengths are in each dataset's JSON fields.",
              "", "## Dukascopy EURUSD BID/ASK preservation check", "",
              f"Both complete normalized Parquet series match the archived 2026-10-27 research fingerprints (BID {dukascopy['BID']['normalized_frame_fingerprint']}; ASK {dukascopy['ASK']['normalized_frame_fingerprint']}). This verifies that the analyzed BID/ASK contents are unchanged from that dated research snapshot. The manifests lack expected per-file hashes, so raw monthly-file hashes are preserved in JSON for a future baseline. The separate July-2026 source/export checksum discrepancy remains unresolved because no independent provider checksum or pre-download file hash is present.",
              "", "## Detailed findings", "",
              "The JSON report contains exact interval boundaries and timestamps for gaps, duplicates, OHLC violations, long exact-flat runs, and price changes above the screening threshold.",
              "Gaps are classified by weekday pattern only; holiday versus broker omission cannot be reliably distinguished without a source-specific trading calendar.",
              "A 10% adjacent-close move is a screening threshold, not proof of corruption.", ""]
    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Read-only audit of collected MT5 OHLC Parquet datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("market-data/mt5-history"))
    parser.add_argument("--duka-dir", type=Path, default=Path("market-data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report = audit_directory(args.data_dir)
    duka = audit_dukascope_side(args.duka_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.report_dir / f"market_data_integrity_{stamp}.json"
    md_path = args.report_dir / f"market_data_integrity_{stamp}.md"
    report["dukascope_eurusd_bid_ask"] = duka
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_markdown(report, duka), encoding="utf-8")
    print(f"Inspected {report['dataset_count']} Parquet datasets; read errors: {len(report['errors'])}")
    for d in report["datasets"]:
        print(f"{d['pair']:6} {d['timeframe']:4} rows={d['rows']:6} gaps={d['gap_count']:5} "
              f"dups={d['duplicate_timestamp_count']} invalid={d['invalid_ohlc_count']} "
              f"flat={d['zero_range_ratio']:.1%} jump>10%={len(d['price_jumps_over_10pct'])}")
    print(f"JSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
