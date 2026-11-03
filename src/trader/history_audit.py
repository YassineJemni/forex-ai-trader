"""Read-only MT5 historical candle coverage audit CLI."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import pandas as pd

from .config import DEFAULT_PAIRS, TIMEFRAME_MINUTES, AppConfig
from .mt5_adapter import MT5DataAdapter, MT5Error

DEFAULT_TIMEFRAMES = ("M5", "M15", "H1", "H4")
CSV_FIELDS = (
    "pair", "broker_symbol", "timeframe", "server_timezone", "history_limit",
    "oldest_candle_utc", "newest_candle_utc", "total_candles", "coverage_days",
    "coverage_years", "missing_interval_count", "missing_intervals",
    "duplicate_count", "invalid_timestamp_count", "invalid_ohlc_count",
    "data_quality_status", "history_status", "history_limit_reached", "error",
    "limitations", "pages_fetched", "max_pages_reached", "request_timeout_exceeded_count", "elapsed_seconds",
)


def _valid_ohlc(row: pd.Series) -> bool:
    try:
        o, h, low, close = (float(row[k]) for k in ("open", "high", "low", "close"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return all(pd.notna(v) and 0 < v < float("inf") for v in (o, h, low, close)) and h >= max(o, close, low) and low <= min(o, close)


def _serialize_time(value: Any) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _audit_rows(adapter: MT5DataAdapter, pair: str, timeframe: str,
                history_limit: int, chunk_size: int, max_pages: int,
                request_timeout_seconds: float) -> dict[str, Any]:
    item_started = time.monotonic()
    row: dict[str, Any] = {
        "pair": pair, "broker_symbol": None, "timeframe": timeframe,
        "server_timezone": adapter.server_timezone, "history_limit": history_limit,
        "oldest_candle_utc": None, "newest_candle_utc": None,
        "total_candles": 0, "coverage_days": 0.0, "coverage_years": 0.0,
        "missing_interval_count": 0, "missing_intervals": [],
        "duplicate_count": 0, "invalid_timestamp_count": 0,
        "invalid_ohlc_count": 0, "data_quality_status": "no_data",
        "history_status": "unavailable", "history_limit_reached": False,
        "error": None, "limitations": [],
        "pages_fetched": 0, "max_pages_reached": False,
        "request_timeout_exceeded_count": 0, "elapsed_seconds": 0.0,
    }
    chunks: list[pd.DataFrame] = []
    try:
        symbol = adapter.resolve_symbol(pair)
        row["broker_symbol"] = symbol
    except MT5Error as exc:
        row["history_status"] = "unavailable" if "symbol" in str(exc).lower() or "resolve" in str(exc).lower() else "api_error"
        row["error"] = str(exc)
        return row
    except Exception as exc:
        row["history_status"] = "api_error"
        row["error"] = f"Unexpected MT5 symbol-resolution error: {exc}"
        return row

    fetched = 0
    while fetched < history_limit and row["pages_fetched"] < max_pages:
        amount = min(chunk_size, history_limit - fetched)
        request_started = time.monotonic()
        try:
            chunk = adapter.history_chunk(symbol, timeframe, start_pos=fetched + 1, count=amount)
        except MT5Error as exc:
            row["history_status"] = "api_error"
            row["error"] = str(exc)
            break
        except Exception as exc:
            row["history_status"] = "api_error"
            row["error"] = f"Unexpected MT5 history API error: {exc}"
            break
        request_elapsed = time.monotonic() - request_started
        row["pages_fetched"] += 1
        if request_elapsed > request_timeout_seconds:
            row["request_timeout_exceeded_count"] += 1
            row["limitations"].append(
                f"History request took {request_elapsed:.2f}s, exceeding the {request_timeout_seconds:.2f}s budget. "
                "MT5 Python calls are synchronous and cannot be safely interrupted; the warning is detected after return."
            )
        if chunk.empty:
            break
        chunks.append(chunk)
        fetched += len(chunk)
        if len(chunk) < amount:
            break
    data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    row["total_candles"] = len(data)
    row["history_limit_reached"] = fetched >= history_limit
    row["max_pages_reached"] = row["pages_fetched"] >= max_pages and fetched < history_limit
    if row["max_pages_reached"]:
        row["limitations"].append(f"Maximum page count ({max_pages}) reached; additional history was not requested.")
    if row["history_limit_reached"]:
        row["limitations"].append("Requested history limit reached; older terminal history may exist.")
    try:
        info = adapter.terminal_info()
        max_bars = getattr(info, "maxbars", None) if info is not None else None
        if max_bars:
            row["limitations"].append(f"MT5 terminal reports maxbars={max_bars}; terminal history settings may cap availability.")
    except MT5Error as exc:
        row["limitations"].append(f"Could not inspect terminal history cap: {exc}")
    row["limitations"].append("MT5 exposes terminal-cached history subject to broker feed and terminal Max. bars in chart settings; API results cannot establish the broker's full archive.")
    if data.empty:
        if row["history_status"] != "api_error":
            row["history_status"] = "unavailable"
        row["data_quality_status"] = "no_data"
        row["elapsed_seconds"] = round(time.monotonic() - item_started, 3)
        return row
    if row["history_status"] != "api_error":
        row["history_status"] = "available" if len(data) else "unavailable"
    if "time" not in data:
        row["history_status"] = "api_error"
        row["error"] = "MT5 history response has no time column"
        row["data_quality_status"] = "invalid"
        row["invalid_timestamp_count"] = len(data)
        row["elapsed_seconds"] = round(time.monotonic() - item_started, 3)
        return row
    try:
        times = adapter._mt5_candle_times_to_utc(data["time"])
        valid_time = times.notna()
    except (ValueError, TypeError, OverflowError):
        times = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
        valid_time = pd.Series(False, index=data.index)
        # Count bad values individually so one malformed stamp doesn't hide the rest.
        for ix, value in data["time"].items():
            try:
                times.loc[ix] = adapter._mt5_candle_times_to_utc(pd.Series([value])).iloc[0]
                valid_time.loc[ix] = True
            except (ValueError, TypeError, OverflowError):
                pass
    row["invalid_timestamp_count"] = int((~valid_time).sum())
    usable = data.loc[valid_time].copy()
    usable["_utc_time"] = times.loc[valid_time].array
    row["duplicate_count"] = int(usable["_utc_time"].duplicated(keep="first").sum())
    if not usable.empty:
        ordered_times = usable["_utc_time"].sort_values(kind="stable").reset_index(drop=True)
        oldest, newest = ordered_times.iloc[0], ordered_times.iloc[-1]
        row["oldest_candle_utc"], row["newest_candle_utc"] = _serialize_time(oldest), _serialize_time(newest)
        seconds = (newest - oldest).total_seconds()
        row["coverage_days"] = round(seconds / 86400, 6)
        row["coverage_years"] = round(seconds / (86400 * 365.2425), 6)
        expected = TIMEFRAME_MINUTES[timeframe] * 60
        unique_times = ordered_times.drop_duplicates().reset_index(drop=True)
        gaps = []
        for left, right in zip(unique_times.iloc[:-1], unique_times.iloc[1:]):
            delta = (right - left).total_seconds()
            if delta > expected:
                gaps.append({"from_utc": _serialize_time(left), "to_utc": _serialize_time(right),
                             "elapsed_minutes": round(delta / 60, 3),
                             "estimated_missing_candles": max(0, int(delta // expected) - 1)})
        row["missing_intervals"] = gaps
        row["missing_interval_count"] = len(gaps)
    row["invalid_ohlc_count"] = int(sum(not _valid_ohlc(r) for _, r in data.iterrows()))
    if row["history_status"] != "api_error":
        row["history_status"] = "available" if not data.empty else "unavailable"
    row["data_quality_status"] = "valid" if not any((row["invalid_timestamp_count"], row["duplicate_count"], row["invalid_ohlc_count"])) else "issues"
    # Gaps reflect weekends/holidays and outages alike; MT5 alone cannot classify them.
    if row["missing_interval_count"]:
        row["limitations"].append("Gap intervals include weekends, holidays, broker outages, and possible missing bars; they are not classified.")
    row["elapsed_seconds"] = round(time.monotonic() - item_started, 3)
    return row


def audit_history(adapter: MT5DataAdapter, pairs: Iterable[str] = DEFAULT_PAIRS,
                  timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
                  history_limit: int = 100_000, chunk_size: int = 1_000,
                  max_pages: int = 10, request_timeout_seconds: float = 15.0,
                  progress: Any | None = None,
                  on_result: Any | None = None) -> list[dict[str, Any]]:
    if history_limit < 1 or chunk_size < 1 or max_pages < 1 or request_timeout_seconds <= 0:
        raise ValueError("history_limit, chunk_size, max_pages and request timeout must be positive")
    pairs, timeframes = tuple(pairs), tuple(x.upper() for x in timeframes)
    unknown = set(timeframes) - set(TIMEFRAME_MINUTES)
    if unknown:
        raise ValueError(f"Unsupported timeframes: {sorted(unknown)}")
    results = []
    total = len(pairs) * len(timeframes)
    for pair in pairs:
        for timeframe in timeframes:
            index = len(results) + 1
            if progress:
                progress("start", index, total, pair.upper(), timeframe, None)
            started = time.monotonic()
            result = _audit_rows(adapter, pair.upper(), timeframe, history_limit,
                                 chunk_size, max_pages, request_timeout_seconds)
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            results.append(result)
            if on_result:
                on_result(results)
            if progress:
                progress("done", index, total, pair.upper(), timeframe, result)
    return results


def _exclusive_write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    base, suffix = path.stem, path.suffix
    candidate, index = path, 1
    while True:
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{base}_{index}{suffix}")
            index += 1


def write_reports(results: list[dict[str, Any]], output_dir: str | Path,
                  generated_at: datetime | None = None,
                  paths: tuple[Path, Path] | None = None) -> tuple[Path, Path]:
    generated_at = generated_at or datetime.now(timezone.utc)
    stamp = generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_dir)
    json_path = root / f"mt5_history_coverage_{stamp}.json"
    csv_path = root / f"mt5_history_coverage_{stamp}.csv"
    payload = {"generated_at_utc": _serialize_time(generated_at.astimezone(timezone.utc)),
               "timestamp_interpretation": "MT5 epoch seconds interpreted as wall-clock time in configured server_timezone, then converted to UTC.",
               "results": results}
    json_content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        flat = dict(row)
        flat["missing_intervals"] = json.dumps(flat["missing_intervals"], separators=(",", ":"))
        flat["limitations"] = json.dumps(flat["limitations"], separators=(",", ":"))
        writer.writerow(flat)
    if paths is None:
        json_out = _exclusive_write(json_path, json_content)
        csv_out = _exclusive_write(csv_path, buffer.getvalue())
    else:
        json_out, csv_out = paths
        _replace_owned(json_out, json_content)
        _replace_owned(csv_out, buffer.getvalue())
    return json_out, csv_out


def _replace_owned(path: Path, content: str) -> None:
    """Atomically refresh only report paths created by this running audit."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _print_report(results: list[dict[str, Any]]) -> None:
    headers = ("PAIR", "TF", "BROKER SYMBOL", "CANDLES", "OLDEST UTC", "NEWEST UTC", "DAYS", "GAPS", "DUPS", "OHLC", "STATUS")
    rows = [[str(r.get("pair")), str(r.get("timeframe")), str(r.get("broker_symbol") or "-"),
             str(r["total_candles"]), str(r["oldest_candle_utc"] or "-"),
             str(r["newest_candle_utc"] or "-"), f'{r["coverage_days"]:.2f}',
             str(r["missing_interval_count"]), str(r["duplicate_count"]),
             str(r["invalid_ohlc_count"]), f'{r["history_status"]}/{r["data_quality_status"]}'] for r in results]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for r in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(r)))
    for result in results:
        if result["error"]:
            print(f'ERROR {result["pair"]} {result["timeframe"]}: {result["error"]}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only MT5 historical candle coverage audit")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS), help="Comma-separated symbols")
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES), help="Comma-separated MT5 timeframes")
    parser.add_argument("--history-limit", type=int, default=100_000, help="Maximum completed candles queried per pair/timeframe")
    parser.add_argument("--chunk-size", type=int, default=1_000, help="Candles per paginated MT5 request")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum MT5 history requests per pair/timeframe")
    parser.add_argument("--request-timeout", type=float, default=15.0, help="Per-request time budget in seconds (reported after return; MT5 calls cannot be safely interrupted)")
    parser.add_argument("--output-dir", default="reports/mt5-history-audit")
    parser.add_argument("--terminal-path", default=None)
    parser.add_argument("--server-timezone", default=None,
                        help="Overrides MT5_SERVER_TIMEZONE (default: configured MetaQuotes-Demo offset Etc/GMT-2)")
    args = parser.parse_args(argv)
    config = AppConfig.from_env()
    adapter = MT5DataAdapter(terminal_path=args.terminal_path or config.mt5_terminal_path,
                             server_timezone=args.server_timezone or config.mt5_server_timezone,
                             max_quote_age_seconds=config.max_quote_age_seconds)
    report_paths: tuple[Path, Path] | None = None
    generated_at = datetime.now(timezone.utc)

    def show_progress(event: str, index: int, total: int, pair: str,
                      timeframe: str, result: dict[str, Any] | None) -> None:
        if event == "start":
            print(f"[{index}/{total}] START {pair} {timeframe}: elapsed=0.00s candles=pending; "
                  f"requesting up to {args.max_pages} page(s) x {args.chunk_size} candles", flush=True)
        else:
            assert result is not None
            print(f"[{index}/{total}] DONE  {pair} {timeframe}: elapsed={result['elapsed_seconds']:.2f}s, "
                  f"candles={result['total_candles']}, "
                  f"{result['history_status']}/{result['data_quality_status']}", flush=True)

    def save_partial(results: list[dict[str, Any]]) -> None:
        nonlocal report_paths
        report_paths = write_reports(results, args.output_dir, generated_at, report_paths)

    try:
        adapter.connect()
        results = audit_history(adapter, args.pairs.split(","), args.timeframes.split(","),
                                args.history_limit, args.chunk_size, args.max_pages,
                                args.request_timeout, progress=show_progress, on_result=save_partial)
    except MT5Error as exc:
        print(f"MT5 connection error: {exc}")
        return 2
    finally:
        adapter.close()
    _print_report(results)
    if report_paths is None:
        report_paths = write_reports(results, args.output_dir, generated_at)
    json_path, csv_path = report_paths
    print(f"\nPartial/final JSON report: {json_path}\nPartial/final CSV report:  {csv_path}")
    return 1 if any(row["history_status"] == "api_error" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
