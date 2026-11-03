import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.research import mission10_broker_calibration as m10
from src.trader.mt5_adapter import MT5Error


class FakeAdapter:
    server_timezone = "Etc/GMT-2"

    def __init__(self, *, fail=False):
        self.fail = fail

    def resolve_symbol(self, pair):
        return pair

    def raw_tick(self, pair):
        if self.fail:
            raise MT5Error("fake tick failure")
        return SimpleNamespace(time=1793700000, time_msc=1793700000000,
                               bid=1.1, ask=1.1001)

    def symbol_info(self, pair):
        return SimpleNamespace(point=0.00001, digits=5, spread=10, _asdict=lambda: {})

    def normalize_tick_timestamp(self, tick, pair):
        return datetime.fromtimestamp(tick.time, timezone.utc)

    def quote_from_tick(self, pair, tick):
        return SimpleNamespace(bid=tick.bid, ask=tick.ask)


def test_pip_size_handles_jpy_and_non_jpy_pairs():
    assert m10.pip_size("USDJPY") == 0.01
    assert m10.pip_size("EURUSD") == 0.0001
    with pytest.raises(ValueError):
        m10.pip_size("EUR")


def test_collection_appends_observations_and_checkpoint(tmp_path):
    history = tmp_path / "samples.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    messages = []
    result = m10.collect_spreads(FakeAdapter(), ["EURUSD"], duration_minutes=1 / 60,
                                 interval_seconds=1, output_path=history,
                                 checkpoint_path=checkpoint, progress=lambda x, **kw: messages.append(x))
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    saved_checkpoint = json.loads(checkpoint.read_text())
    assert result["rows_added"] == len(rows) == 1
    assert rows[0]["validation_status"] == "valid"
    assert rows[0]["timezone_configuration"] == "Etc/GMT-2"
    assert saved_checkpoint["completed_sweeps"] == 1
    assert any("1/1 EURUSD" in message for message in messages)


def test_collection_records_api_errors_and_continues(tmp_path):
    history = tmp_path / "samples.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    result = m10.collect_spreads(FakeAdapter(fail=True), ["EURUSD", "GBPUSD"],
                                 duration_minutes=1 / 60, interval_seconds=1,
                                 output_path=history, checkpoint_path=checkpoint)
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert result["rows_added"] == 2
    assert len(result["errors"]) == 2
    assert all(row["validation_status"] == "api_error" for row in rows)
    assert json.loads(checkpoint.read_text())["completed_sweeps"] == 1


def test_resume_requires_both_existing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="--resume"):
        m10.collect_spreads(FakeAdapter(), ["EURUSD"], duration_minutes=1,
                            interval_seconds=1, output_path=tmp_path / "a.jsonl",
                            checkpoint_path=tmp_path / "b.json", resume=True)


def test_spread_statistics_handles_error_only_rows():
    rows = [{"pair": "EURUSD", "observation_utc": "2026-11-03T10:00:00+00:00",
             "primary_session_utc": "London", "validation_status": "api_error",
             "sample_duplicate": False}]
    stats = m10.spread_statistics(rows)
    assert stats[0]["valid_unique_sample_count"] == 0
    assert stats[0]["median_spread_pips"] is None
    assert stats[0]["distribution_status"] == "insufficient_sample_or_coverage"


def test_duplicate_quotes_are_marked_and_excluded_from_unique_distribution(tmp_path):
    history = tmp_path / "samples.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    m10.collect_spreads(FakeAdapter(), ["EURUSD"], duration_minutes=2 / 60,
                        interval_seconds=1, output_path=history,
                        checkpoint_path=checkpoint)
    rows = m10.load_jsonl(history)
    assert len(rows) == 2
    assert rows[0]["sample_duplicate"] is False
    assert rows[1]["sample_duplicate"] is True
    stats = m10.spread_statistics(rows)
    assert stats[0]["valid_unique_sample_count"] == 1
    assert stats[0]["duplicate_count"] == 1
