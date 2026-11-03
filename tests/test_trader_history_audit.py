from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from src.trader.history_audit import audit_history, write_reports
from src.trader.mt5_adapter import MT5DataAdapter


class HistoryMT5:
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4 = 1, 5, 15, 60, 240

    def __init__(self, data):
        self.data = data
        self.calls = []

    def initialize(self, **kwargs): return True
    def shutdown(self): pass
    def terminal_info(self): return SimpleNamespace(maxbars=100_000)
    def last_error(self): return (1, "Success")
    def symbol_select(self, symbol, enabled): return True
    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        self.calls.append((symbol, timeframe, start, count))
        offset = start - 1
        return self.data[offset:offset + count]


def _rates(times):
    return pd.DataFrame({"time": [int(t.timestamp()) for t in times],
                         "open": [1.1] * len(times), "high": [1.2] * len(times),
                         "low": [1.0] * len(times), "close": [1.15] * len(times)})


def test_audit_paginates_and_reports_coverage_gaps_duplicates_and_ohlc():
    times = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:05Z",
                            "2026-01-01T00:05Z", "2026-01-01T00:20Z"])
    data = _rates(times).reset_index(drop=True)
    data.loc[3, "high"] = 0.9
    mt5 = HistoryMT5(data)
    adapter = MT5DataAdapter(terminal=mt5)
    result = audit_history(adapter, ["EURUSD"], ["M5"], history_limit=4, chunk_size=2)[0]
    assert result["broker_symbol"] == "EURUSD"
    assert result["total_candles"] == 4
    assert result["oldest_candle_utc"] == "2026-01-01T00:00:00Z"
    assert result["newest_candle_utc"] == "2026-01-01T00:20:00Z"
    assert result["coverage_days"] > 0
    assert result["duplicate_count"] == 1
    assert result["missing_interval_count"] == 1
    assert result["invalid_ohlc_count"] == 1
    assert result["history_limit_reached"]
    assert [c[2] for c in mt5.calls] == [1, 3]


def test_audit_distinguishes_empty_history_and_symbol_unavailable():
    empty = MT5DataAdapter(terminal=HistoryMT5(pd.DataFrame()))
    assert audit_history(empty, ["EURUSD"], ["H1"])[0]["history_status"] == "unavailable"
    mt5 = HistoryMT5(pd.DataFrame())
    mt5.symbol_select = lambda symbol, enabled: False
    missing = audit_history(MT5DataAdapter(terminal=mt5), ["EURUSD"], ["H1"])[0]
    assert missing["history_status"] == "unavailable"
    assert missing["error"]


def test_server_timezone_converts_epoch_and_report_writer_never_overwrites(tmp_path):
    # 02:00 server wall time at fixed UTC+2 corresponds to midnight UTC.
    local = pd.to_datetime(["2026-01-01T02:00:00Z", "2026-01-01T02:05:00Z"])
    adapter = MT5DataAdapter(terminal=HistoryMT5(_rates(local)), server_timezone="Etc/GMT-2")
    result = audit_history(adapter, ["EURUSD"], ["M5"], history_limit=2)[0]
    assert result["oldest_candle_utc"] == "2026-01-01T00:00:00Z"
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = write_reports([result], tmp_path, stamp)
    second = write_reports([result], tmp_path, stamp)
    assert first[0].exists() and first[1].exists()
    assert all(path != other for path, other in zip(first, second))
    assert "Etc/GMT-2" in first[0].read_text()
    assert "EURUSD" in first[1].read_text()


def test_api_none_response_is_reported_as_api_error():
    mt5 = HistoryMT5(pd.DataFrame())
    mt5.copy_rates_from_pos = lambda *args: None
    result = audit_history(MT5DataAdapter(terminal=mt5), ["EURUSD"], ["H1"])[0]
    assert result["history_status"] == "api_error"
    assert "history API error" in result["error"]


def test_progress_and_partial_reports_are_emitted_after_each_combination(tmp_path):
    times = pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC")
    adapter = MT5DataAdapter(terminal=HistoryMT5(_rates(times)))
    events = []
    report_paths = None

    def progress(event, index, total, pair, timeframe, result):
        events.append((event, index, pair, timeframe, result))

    def checkpoint(results):
        nonlocal report_paths
        report_paths = write_reports(results, tmp_path, datetime(2026, 1, 1, tzinfo=timezone.utc), report_paths)
        import json
        saved = json.loads(report_paths[0].read_text())
        assert len(saved["results"]) == len(results)

    results = audit_history(adapter, ["EURUSD", "GBPUSD"], ["M5"],
                            history_limit=2, chunk_size=1, max_pages=2,
                            progress=progress, on_result=checkpoint)
    assert len(results) == 2
    assert [e[0] for e in events] == ["start", "done", "start", "done"]
    assert all(e[4]["elapsed_seconds"] >= 0 for e in events if e[0] == "done")
    assert report_paths and "GBPUSD" in report_paths[1].read_text()


def test_request_error_is_recorded_and_audit_continues_to_next_timeframe():
    times = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    mt5 = HistoryMT5(_rates(times))
    def broken_on_m5(symbol, timeframe, start, count):
        if timeframe == mt5.TIMEFRAME_M5:
            raise RuntimeError("terminal busy")
        offset = start - 1
        return mt5.data[offset:offset + count]
    mt5.copy_rates_from_pos = broken_on_m5
    result = audit_history(MT5DataAdapter(terminal=mt5), ["EURUSD"], ["M5", "H1"],
                           history_limit=2, chunk_size=2)
    assert result[0]["history_status"] == "api_error"
    assert "terminal busy" in result[0]["error"]
    assert result[1]["history_status"] == "available"


def test_max_pages_bounds_requests_and_marks_partial_coverage():
    times = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
    mt5 = HistoryMT5(_rates(times))
    result = audit_history(MT5DataAdapter(terminal=mt5), ["EURUSD"], ["M5"],
                           history_limit=100, chunk_size=2, max_pages=2)[0]
    assert result["total_candles"] == 4
    assert result["pages_fetched"] == 2
    assert result["max_pages_reached"]
    assert len(mt5.calls) == 2


def test_request_timeout_budget_is_reported_after_synchronous_call_returns():
    import time
    times = pd.date_range("2026-01-01", periods=1, freq="5min", tz="UTC")
    mt5 = HistoryMT5(_rates(times))
    original = mt5.copy_rates_from_pos
    def slow_call(*args):
        time.sleep(0.02)
        return original(*args)
    mt5.copy_rates_from_pos = slow_call
    result = audit_history(MT5DataAdapter(terminal=mt5), ["EURUSD"], ["M5"],
                           history_limit=1, chunk_size=1, request_timeout_seconds=0.001)[0]
    assert result["request_timeout_exceeded_count"] == 1
    assert any("cannot be safely interrupted" in note for note in result["limitations"])
