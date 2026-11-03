from datetime import datetime, timezone

import pandas as pd

from src.trader.data_collection import collect_incremental


class FakeHistoryAdapter:
    def __init__(self, candles, *, fail_at=None):
        self.data = candles
        self.resolutions = []
        self.calls = []
        self.fail_at = fail_at
        self.server_timezone = "UTC"

    def resolve_symbol(self, symbol):
        self.resolutions.append(symbol)
        return f"{symbol}.x"

    def candles(self, symbol, timeframe, count):
        assert symbol.endswith((".x", ".pro"))
        assert timeframe == "M5"
        return self.data.copy()

    def _mt5_candle_times_to_utc(self, values):
        return pd.to_datetime(values, unit="s", utc=True)

    def history_chunk(self, symbol, timeframe, start_pos, count):
        self.calls.append((symbol, timeframe, start_pos, count))
        if start_pos == self.fail_at:
            raise RuntimeError("simulated history error")
        # MT5 position 1 is newest; response rows are chronological.
        end = max(0, len(self.data) - start_pos + 1)
        begin = max(0, end - count)
        page = self.data.iloc[begin:end].copy()
        page["time"] = pd.to_datetime(page.time, utc=True).astype("int64") // 1_000_000
        return page.reset_index(drop=True)


def candles(count=5, *, invalid=False):
    times = pd.date_range("2026-01-01", periods=count, freq="10min", tz="UTC")
    high = [1.2] * count
    if invalid:
        high[-1] = 1.0
    return pd.DataFrame({"time": times, "open": [1.1] * count,
                         "high": high, "low": [1.0] * count,
                         "close": [1.15] * count, "spread": [2] * count})


def test_incremental_collection_persists_provenance_and_does_not_fill_gaps(tmp_path):
    adapter = FakeHistoryAdapter(candles())
    first = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                data_dir=tmp_path, bars=10)[0]
    assert first["status"] in {"updated", "page_limit"}
    assert first["quality"]["missing_interval_gaps"] == 4
    saved = pd.read_parquet(tmp_path / "EURUSD_M5.parquet")
    assert str(saved.time.dt.tz) == "UTC"
    assert saved.source.eq("MetaTrader5 Python API").all()
    assert saved.collected_at_utc.notna().all()

    adapter.data = candles(6)
    second = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                 data_dir=tmp_path, bars=10)[0]
    saved = pd.read_parquet(tmp_path / "EURUSD_M5.parquet")
    assert second["duplicate_overlap_rows_replaced"] == 5
    assert second["rows_after_update"] == 6
    assert len(saved) == 6
    assert (tmp_path / "EURUSD_M5.provenance.json").exists()


def test_invalid_update_preserves_existing_collected_data(tmp_path):
    adapter = FakeHistoryAdapter(candles())
    collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                        data_dir=tmp_path, bars=10)
    path = tmp_path / "EURUSD_M5.parquet"
    original = pd.read_parquet(path)
    adapter.data = candles(6, invalid=True)
    report = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                 data_dir=tmp_path, bars=10)[0]
    assert report["status"] == "partial"
    assert report["last_run_errors"]
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)


def test_broker_symbol_mapping_is_respected(tmp_path):
    adapter = FakeHistoryAdapter(candles())
    report = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                 data_dir=tmp_path, bars=10,
                                 symbol_map={"EURUSD": "EURUSD.pro"})[0]
    assert report["terminal_symbol"] == "EURUSD.pro"
    assert adapter.resolutions == []


def test_bounded_pagination_and_resume_extend_older_history_without_duplicates(tmp_path):
    adapter = FakeHistoryAdapter(candles(8))
    progress = []
    first = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                data_dir=tmp_path, chunk_size=2, max_pages=2,
                                progress=progress.append)[0]
    assert first["status"] == "page_limit"
    assert first["received_candles"] == 4
    assert any("PAGE EURUSD M5" in line for line in progress)
    saved_path = tmp_path / "EURUSD_M5.parquet"
    first_saved = pd.read_parquet(saved_path)
    assert len(first_saved) == 4
    assert first["next_start_pos"] == 5

    second = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                 data_dir=tmp_path, chunk_size=2, max_pages=1,
                                 resume=True)[0]
    saved = pd.read_parquet(saved_path)
    assert len(saved) == 6
    assert second["next_start_pos"] == 7
    assert second["duplicate_overlap_rows_replaced"] == 0
    assert saved.time.is_monotonic_increasing and saved.time.is_unique


def test_invalid_batch_is_rejected_and_valid_stored_data_is_preserved(tmp_path):
    adapter = FakeHistoryAdapter(candles(4))
    collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                        data_dir=tmp_path, chunk_size=4, max_pages=1)
    path = tmp_path / "EURUSD_M5.parquet"
    original = pd.read_parquet(path)
    broken = candles(6)
    broken.loc[0, "high"] = 1.0
    adapter.data = broken
    report = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                 data_dir=tmp_path, chunk_size=2, max_pages=1)[0]
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert report["status"] == "partial"
    assert report["last_run_errors"]
    run = report["collection_runs"][-1]
    assert run["invalid_rows_rejected"] == 2
    assert len(run["pages"]) == 1 and run["pages"][0]["error"]


def test_failed_page_keeps_resume_cursor_for_checkpoint_recovery(tmp_path):
    adapter = FakeHistoryAdapter(candles(8), fail_at=3)
    first = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                data_dir=tmp_path, chunk_size=2, max_pages=3)[0]
    assert first["status"] == "partial"
    assert first["next_start_pos"] == 3
    assert len(pd.read_parquet(tmp_path / "EURUSD_M5.parquet")) == 2

    adapter.fail_at = None
    recovered = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5"],
                                    data_dir=tmp_path, chunk_size=2, max_pages=1,
                                    resume=True)[0]
    saved = pd.read_parquet(tmp_path / "EURUSD_M5.parquet")
    assert len(saved) == 4
    assert recovered["status"] in {"updated", "page_limit"}
    assert saved.time.is_unique


def test_slow_mt5_request_is_recorded_checkpointed_and_next_combo_continues(tmp_path):
    import time
    adapter = FakeHistoryAdapter(candles(6))
    original = adapter.history_chunk
    def slow_history(symbol, timeframe, start_pos, count):
        if timeframe == "H4":
            time.sleep(0.02)
        return original(symbol, timeframe, start_pos, count)
    adapter.history_chunk = slow_history
    reports = collect_incremental(adapter, pairs=["EURUSD"], timeframes=["M5", "H4"],
                                  data_dir=tmp_path, chunk_size=2, max_pages=3,
                                  request_timeout_seconds=0.001, progress=None)
    # The 1 ms request budget may also be exceeded by an undelayed request on
    # a loaded CI/Windows host; either status preserves the checkpoint.
    assert reports[0]["status"] in {"page_limit", "partial"}
    assert reports[1]["status"] == "partial"
    assert reports[1]["collection_runs"][-1]["slow_requests"]
    assert (tmp_path / "EURUSD_M5.parquet").exists()
    assert (tmp_path / "EURUSD_H4.parquet").exists()
