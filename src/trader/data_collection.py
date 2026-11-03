"""Bounded, resumable, read-only MT5 historical collection into Parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.research.data import validate_ohlc_prices
from .config import DEFAULT_PAIRS, TIMEFRAME_MINUTES
from .mt5_adapter import MT5DataAdapter, MT5Error


def _safe_name(value: str) -> str:
    compact = "".join(ch for ch in value.upper() if ch.isalnum())
    if not compact:
        raise ValueError("symbol/timeframe must contain alphanumeric characters")
    return compact


def _fingerprint(data: pd.DataFrame) -> str:
    columns = ["time", "open", "high", "low", "close"]
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(data[columns], index=False).values.tobytes())
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".history-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".provenance-", suffix=".json", dir=path.parent)
    os.close(fd)
    try:
        Path(temp).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _quality(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    ordered = frame.sort_values("time", kind="stable")
    intervals = ordered.time.diff().dropna()
    expected = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    return {
        "rows": int(len(frame)),
        "oldest_timestamp_utc": ordered.time.iloc[0].isoformat() if len(frame) else None,
        "newest_timestamp_utc": ordered.time.iloc[-1].isoformat() if len(frame) else None,
        "duplicate_timestamps": int(frame.time.duplicated().sum()),
        "nonpositive_intervals": int((intervals <= pd.Timedelta(0)).sum()),
        "missing_interval_gaps": int((intervals > expected).sum()),
        "invalid_ohlc_rows": 0,
    }


def _configured_timezone(data_dir: str | Path) -> str:
    """Reuse the timezone recorded with existing collections when env is unset."""
    configured = os.getenv("MT5_SERVER_TIMEZONE", "").strip()
    if configured:
        return configured
    timezones = set()
    for manifest in Path(data_dir).glob("*.provenance.json"):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8")).get("configured_server_timezone")
            if value:
                timezones.add(value)
        except (OSError, json.JSONDecodeError):
            continue
    if len(timezones) > 1:
        raise ValueError(f"Conflicting MT5 server timezones in collection manifests: {sorted(timezones)}; set MT5_SERVER_TIMEZONE explicitly")
    return next(iter(timezones), "UTC")


def _normalize_batch(adapter: MT5DataAdapter, raw: pd.DataFrame,
                     timeframe: str, requested: str, terminal_symbol: str,
                     collected_at: str) -> tuple[pd.DataFrame, int, int]:
    if raw.empty:
        return pd.DataFrame(), 0, 0
    if "time" not in raw:
        raise ValueError("MT5 history batch has no time column")
    try:
        data = raw.copy().reset_index(drop=True)
        data["time"] = adapter._mt5_candle_times_to_utc(data["time"])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"MT5 history batch has invalid timestamps: {exc}") from exc
    if data.time.isna().any():
        raise ValueError("MT5 history batch includes invalid timestamps")
    data = data.sort_values("time", kind="stable").reset_index(drop=True)
    duplicates = int(data.time.duplicated(keep=False).sum())
    if duplicates:
        # Identical duplicate rows are safe to collapse. Conflicting duplicates
        # are ambiguous and the whole page is rejected, preserving prior values.
        for _, group in data[data.time.duplicated(keep=False)].groupby("time", sort=False):
            comparable = group.drop(columns=["time"]).reset_index(drop=True)
            if len(comparable.drop_duplicates()) != 1:
                raise ValueError("MT5 history batch contains conflicting duplicate timestamps")
        data = data.drop_duplicates("time", keep="last").reset_index(drop=True)
    data = validate_ohlc_prices(data)
    data["requested_symbol"] = requested
    data["terminal_symbol"] = terminal_symbol
    data["timeframe"] = timeframe
    data["source"] = "MetaTrader5 Python API"
    data["collected_at_utc"] = collected_at
    return data, duplicates, len(raw) - len(data)


def collect_incremental(adapter: MT5DataAdapter, *, pairs: Sequence[str] = DEFAULT_PAIRS,
                        timeframes: Sequence[str] = ("M5", "M15", "H1", "H4"),
                        data_dir: str | Path = "market-data/mt5-history",
                        chunk_size: int = 1_000, max_pages: int = 101,
                        resume: bool = True, symbol_map: Mapping[str, str] | None = None,
                        request_timeout_seconds: float = 60.0,
                        progress: Any | None = print,
                        bars: int | None = None) -> list[dict[str, Any]]:
    """Fetch bounded completed-bar pages and merge validated data by UTC time.

    Position 1 is the newest completed MT5 candle. Existing provenance stores a
    resume cursor for the next older position; each run refreshes a recent page,
    then continues from that cursor. No candles are synthesized or gap-filled.
    ``bars`` is retained as a compatibility alias for one bounded page.
    """
    if bars is not None:
        chunk_size, max_pages = bars, 1
    if chunk_size < 1 or max_pages < 1 or request_timeout_seconds <= 0:
        raise ValueError("chunk_size, max_pages and request timeout must be positive")
    unsupported = set(x.upper() for x in timeframes) - set(TIMEFRAME_MINUTES)
    if unsupported:
        raise ValueError(f"Unsupported timeframes: {sorted(unsupported)}")
    root = Path(data_dir).resolve()
    mapping = {key.upper(): value for key, value in (symbol_map or {}).items()}
    reports: list[dict[str, Any]] = []
    total = len(pairs) * len(timeframes)
    for pair in pairs:
        for timeframe in timeframes:
            requested, timeframe = pair.upper(), timeframe.upper()
            index = len(reports) + 1
            started = time.monotonic()
            if progress:
                progress(f"[{index}/{total}] START {requested} {timeframe}: max {max_pages} page(s) x {chunk_size} candles")
            terminal_symbol = mapping.get(requested)
            stem = f"{_safe_name(requested[:6])}_{_safe_name(timeframe)}"
            path, manifest_path = root / f"{stem}.parquet", root / f"{stem}.provenance.json"
            try:
                terminal_symbol = terminal_symbol or adapter.resolve_symbol(requested)
                prior = pd.read_parquet(path) if path.exists() else pd.DataFrame()
                if not prior.empty:
                    if prior.time.isna().any() or prior.time.duplicated().any():
                        raise ValueError("existing dataset has invalid or duplicate timestamps; refusing to alter it")
                    prior = validate_ohlc_prices(prior)
                old_rows = len(prior)
                previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                terminal_maxbars = None
                try:
                    terminal_info = adapter.terminal_info()
                    terminal_maxbars = getattr(terminal_info, "maxbars", None) if terminal_info is not None else None
                except (MT5Error, AttributeError):
                    pass
                resume_cursor = int(previous_manifest.get("next_start_pos", old_rows + 1))
                exhausted = bool(previous_manifest.get("history_exhausted", False))
                now = datetime.now(timezone.utc).isoformat()
                request_starts = [1]
                if resume and not exhausted and resume_cursor > 1:
                    request_starts = [resume_cursor, 1]
                elif not resume:
                    request_starts = [1]
                # For a new dataset, continue immediately after each successful page.
                if old_rows == 0:
                    request_starts = [1]
                new_batches: list[pd.DataFrame] = []
                request_log: list[dict[str, Any]] = []
                errors: list[str] = []
                slow_requests: list[dict[str, Any]] = []
                duplicate_rows = invalid_rows = requested_count = received_count = 0
                cursor = resume_cursor if resume and old_rows else 1
                seen_starts: set[int] = set()
                while len(request_log) < max_pages:
                    if request_starts:
                        start_pos = request_starts.pop(0)
                    else:
                        if exhausted or cursor in seen_starts:
                            break
                        start_pos = cursor
                    if start_pos in seen_starts:
                        continue
                    seen_starts.add(start_pos)
                    requested_count += chunk_size
                    if progress:
                        progress(f"[{index}/{total}] PAGE {requested} {timeframe}: "
                                 f"request={len(request_log) + 1}/{max_pages} start_pos={start_pos} "
                                 f"requested={chunk_size}")
                    page_started = time.monotonic()
                    raw = None
                    page_record: dict[str, Any] | None = None
                    try:
                        raw = adapter.history_chunk(terminal_symbol, timeframe,
                                                    start_pos=start_pos, count=chunk_size)
                        received_count += len(raw)
                        request_elapsed = time.monotonic() - page_started
                        exceeded_budget = request_elapsed > request_timeout_seconds
                        page_record = {
                            "start_pos": start_pos, "requested_candles": chunk_size,
                            "received_candles": len(raw),
                            "elapsed_seconds": round(request_elapsed, 3),
                            "timeout_budget_seconds": request_timeout_seconds,
                            "timeout_budget_exceeded": exceeded_budget,
                            "error": None,
                        }
                        request_log.append(page_record)
                        if progress:
                            progress(f"[{index}/{total}] PAGE DONE {requested} {timeframe}: "
                                     f"start_pos={start_pos} received={len(raw)} "
                                     f"elapsed={request_elapsed:.2f}s"
                                     f"{' OVER-BUDGET' if exceeded_budget else ''}")
                        if exceeded_budget:
                            slow_requests.append({"start_pos": start_pos,
                                                  "elapsed_seconds": round(request_elapsed, 3),
                                                  "budget_seconds": request_timeout_seconds,
                                                  "limitation": "MT5 call completed synchronously; it cannot be safely interrupted."})
                        if raw.empty:
                            exhausted = True
                            if request_starts:
                                continue
                            break
                        batch, page_duplicates, _ = _normalize_batch(
                            adapter, raw, timeframe, requested, terminal_symbol, now
                        )
                        duplicate_rows += page_duplicates
                        new_batches.append(batch)
                        if start_pos != 1 or old_rows == 0:
                            cursor = start_pos + len(raw)
                        if len(raw) < chunk_size:
                            exhausted = True
                        if exceeded_budget:
                            # Keep this valid page and cursor checkpoint, then
                            # advance to the next pair/timeframe.
                            break
                        if len(raw) < chunk_size:
                            if request_starts:
                                continue
                            break
                        if not resume and old_rows:
                            break
                    except Exception as exc:
                        message = f"start_pos={start_pos}: {type(exc).__name__}: {exc}"
                        errors.append(message)
                        if page_record is not None:
                            page_record["error"] = message
                            page_record["elapsed_seconds"] = round(time.monotonic() - page_started, 3)
                        else:
                            request_log.append({"start_pos": start_pos,
                                                "requested_candles": chunk_size,
                                                "received_candles": 0,
                                                "elapsed_seconds": round(time.monotonic() - page_started, 3),
                                                "error": message})
                        if isinstance(exc, ValueError) and raw is not None:
                            invalid_rows += len(raw)
                        # Keep the resume cursor at the failed page for retry.
                        cursor = start_pos
                        break
                new = pd.concat(new_batches, ignore_index=True) if new_batches else pd.DataFrame()
                all_data = pd.concat([prior, new], ignore_index=True) if not prior.empty and not new.empty else (prior.copy() if not new_batches else new)
                overlap_count = int(all_data.time.duplicated(keep="last").sum()) if not all_data.empty else 0
                if not all_data.empty:
                    all_data = (all_data.sort_values("time", kind="stable")
                                .drop_duplicates("time", keep="last").reset_index(drop=True))
                    # Every incoming page was validated before merge; validate
                    # the complete result before replacing the stored file.
                    all_data = validate_ohlc_prices(all_data)
                    quality = _quality(all_data, timeframe)
                    fingerprint = _fingerprint(all_data)
                    _atomic_parquet(all_data, path)
                else:
                    quality = {"rows": 0, "oldest_timestamp_utc": None,
                               "newest_timestamp_utc": None, "duplicate_timestamps": 0,
                               "nonpositive_intervals": 0, "missing_interval_gaps": 0,
                               "invalid_ohlc_rows": 0}
                    fingerprint = None
                prior_runs = previous_manifest.get("collection_runs", [])
                run = {
                    "retrieval_time_utc": now, "requested_candles": requested_count,
                    "received_candles": received_count, "accepted_candles": len(new),
                    "pages": request_log, "errors": errors,
                    "slow_requests": slow_requests,
                    "duplicate_rows_in_batches": duplicate_rows,
                    "overlap_timestamp_rows_merged": overlap_count,
                    "invalid_rows_rejected": invalid_rows,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "resume_enabled": resume,
                }
                next_pos = cursor
                if not resume:
                    next_pos = max(1, old_rows + 1)
                manifest = {
                    "schema_version": 2, "requested_symbol": requested,
                    "terminal_symbol": terminal_symbol,
                    "broker_symbol": terminal_symbol,
                    "configured_server_timezone": getattr(adapter, "server_timezone", "unknown"),
                    "timeframe": timeframe, "source": "MetaTrader5 Python API",
                    "collection_time_utc": now, "rows_before_update": old_rows,
                    "new_rows_received": len(new), "duplicate_overlap_rows_replaced": overlap_count,
                    "rows_after_update": int(len(all_data)), "data_fingerprint_sha256": fingerprint,
                    "quality": quality, "missing_intervals_are_not_filled": True,
                    "next_start_pos": next_pos, "history_exhausted": exhausted,
                    "terminal_maxbars": terminal_maxbars,
                    "configured_chunk_size": chunk_size, "configured_max_pages": max_pages,
                    "configured_request_timeout_seconds": request_timeout_seconds,
                    "bounded_request_limit_reached": len(request_log) >= max_pages and not exhausted,
                    "collection_runs": [*prior_runs, run],
                    "last_run_errors": errors,
                }
                _atomic_json(manifest, manifest_path)
                elapsed = time.monotonic() - started
                status = "partial" if errors or slow_requests else (
                    "page_limit" if manifest["bounded_request_limit_reached"] else "updated"
                )
                report = {**manifest, "path": str(path),
                          "status": status,
                          "requested_candles": requested_count,
                          "received_candles": received_count,
                          "elapsed_seconds": round(elapsed, 3)}
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                report = {"requested_symbol": requested, "timeframe": timeframe,
                          "terminal_symbol": terminal_symbol, "status": "error",
                          "error": message,
                          "requested_candles": 0, "received_candles": 0,
                          "elapsed_seconds": round(time.monotonic() - started, 3)}
                # Preserve dataset bytes while still checkpointing failed
                # symbol resolution, unreadable files, or validation errors.
                try:
                    failed_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                    failures = failed_manifest.get("collection_runs", [])
                    failures.append({"retrieval_time_utc": datetime.now(timezone.utc).isoformat(),
                                     "errors": [message], "resume_enabled": resume,
                                     "elapsed_seconds": report["elapsed_seconds"]})
                    failed_manifest.update({
                        "schema_version": max(2, int(failed_manifest.get("schema_version", 2))),
                        "requested_symbol": requested, "terminal_symbol": terminal_symbol,
                        "broker_symbol": terminal_symbol,
                        "configured_server_timezone": getattr(adapter, "server_timezone", "unknown"),
                        "timeframe": timeframe, "source": "MetaTrader5 Python API",
                        "collection_runs": failures, "last_run_errors": [message],
                    })
                    _atomic_json(failed_manifest, manifest_path)
                except Exception as manifest_exc:
                    report["manifest_checkpoint_error"] = str(manifest_exc)
            reports.append(report)
            if progress:
                quality = report.get("quality", {})
                progress(f"[{index}/{total}] DONE {requested} {timeframe}: "
                         f"elapsed={report['elapsed_seconds']:.2f}s requested={report['requested_candles']} "
                         f"received={report['received_candles']} candles="
                         f"{quality.get('rows', 0)} coverage="
                         f"{quality.get('oldest_timestamp_utc', '-') }..{quality.get('newest_timestamp_utc', '-')} "
                         f"status={report['status']}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bounded, resumable completed MT5 candles (read-only)")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--timeframes", default="M5,M15,H1,H4")
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--max-pages", type=int, default=101,
                        help="Maximum history calls per pair/timeframe (hard collection bound)")
    parser.add_argument("--request-timeout", type=float, default=60.0,
                        help="Stop paging a combination if a call exceeds this budget; checked after MT5 returns")
    parser.add_argument("--data-dir", default="market-data/mt5-history")
    parser.add_argument("--symbol-map", help="JSON object mapping canonical pairs to broker symbols")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Continue from each dataset's saved older-history cursor (default: on)")
    args = parser.parse_args()
    symbol_map = json.loads(Path(args.symbol_map).read_text(encoding="utf-8")) if args.symbol_map else {}
    adapter = MT5DataAdapter(terminal_path=os.getenv("MT5_TERMINAL_PATH"),
                             server_timezone=_configured_timezone(args.data_dir))
    def progress(message: str) -> None:
        print(message, flush=True)
    try:
        adapter.connect()
        reports = collect_incremental(
            adapter, pairs=[x.strip().upper() for x in args.pairs.split(",") if x.strip()],
            timeframes=[x.strip().upper() for x in args.timeframes.split(",") if x.strip()],
            data_dir=args.data_dir, chunk_size=args.chunk_size,
            max_pages=args.max_pages, resume=args.resume, symbol_map=symbol_map,
            request_timeout_seconds=args.request_timeout,
            progress=progress,
        )
    except MT5Error as exc:
        print(json.dumps({"status": "connection_error", "error": str(exc)}))
        raise SystemExit(2)
    finally:
        adapter.close()
    for report in reports:
        quality = report.get("quality", {})
        print(json.dumps({
            "pair": report.get("requested_symbol"), "timeframe": report.get("timeframe"),
            "broker_symbol": report.get("broker_symbol", report.get("terminal_symbol")),
            "status": report.get("status"), "candles": quality.get("rows", 0),
            "oldest_utc": quality.get("oldest_timestamp_utc"),
            "newest_utc": quality.get("newest_timestamp_utc"),
            "requested": report.get("requested_candles"),
            "received": report.get("received_candles"),
            "maxbars": report.get("terminal_maxbars"),
            "history_exhausted": report.get("history_exhausted"),
            "error": report.get("error") or report.get("last_run_errors"),
        }, sort_keys=True))
    if any(item["status"] in {"error", "partial"} for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
