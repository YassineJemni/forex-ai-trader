from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.trader.mt5_adapter import MT5DataAdapter, MT5Error


class FakeMT5:
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4 = 1, 5, 15, 60, 240

    def __init__(self, now, *, tick_age=1, rates=None):
        self.now = now
        self.tick_age = tick_age
        self.rates = rates
        self.selected = []
        self.initialized = 0
        self.shutdowns = 0
        self.online = True

    def terminal_info(self):
        return object() if self.online else None

    def initialize(self, **kwargs):
        self.initialized += 1
        return True

    def shutdown(self):
        self.shutdowns += 1

    def last_error(self):
        return (1, "fake error")

    def symbol_select(self, symbol, enabled):
        self.selected.append((symbol, enabled))
        return True

    def symbol_info_tick(self, symbol):
        stamp = int(self.now.timestamp()) - self.tick_age
        return SimpleNamespace(time=stamp, time_msc=stamp * 1000, bid=1.1, ask=1.1001)

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        assert start == 1
        return self.rates

    def positions_get(self):
        return []


def test_reads_fresh_quote_and_never_exposes_order_methods():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    api = FakeMT5(now)
    adapter = MT5DataAdapter(terminal=api, now=lambda: now)
    quote = adapter.quote("EURUSD")
    assert quote.bid == 1.1 and quote.ask == 1.1001
    assert adapter.connected
    assert not hasattr(adapter, "order_send")
    adapter.close()
    assert api.shutdowns == 1


def test_rejects_stale_or_crossed_quotes():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    api = FakeMT5(now, tick_age=300)
    adapter = MT5DataAdapter(terminal=api, now=lambda: now)
    with pytest.raises(MT5Error, match="Stale MT5 quote"):
        adapter.quote("EURUSD")
    api.tick_age = 1
    api.symbol_info_tick = lambda _: SimpleNamespace(time=int(now.timestamp()) - 1,
                                                       time_msc=(int(now.timestamp()) - 1) * 1000,
                                                       bid=1.2, ask=1.1)
    with pytest.raises(MT5Error, match="Invalid or crossed"):
        adapter.quote("EURUSD")


def test_rejects_future_quote_after_timezone_normalization():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    api = FakeMT5(now)
    api.symbol_info_tick = lambda _: SimpleNamespace(
        time=int(now.timestamp()) + 60,
        time_msc=(int(now.timestamp()) + 60) * 1000,
        bid=1.1, ask=1.1001,
    )
    with pytest.raises(MT5Error, match="future"):
        MT5DataAdapter(terminal=api, now=lambda: now).quote("EURUSD")


def test_local_aware_clock_is_normalized_to_utc_without_local_shift():
    now_utc = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    local_clock = now_utc.astimezone(ZoneInfo("Europe/Budapest"))
    adapter = MT5DataAdapter(terminal=FakeMT5(now_utc), now=lambda: local_clock)
    quote = adapter.quote("EURUSD")
    assert quote.time == now_utc.replace(microsecond=0) - pd.Timedelta(seconds=1)
    assert quote.time.tzinfo == timezone.utc


def test_explicit_broker_timezone_corrects_server_wall_time_for_quote_and_candles():
    now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    server_wall_epoch = int(now.timestamp()) + 2 * 60 * 60
    api = FakeMT5(now)
    api.symbol_info_tick = lambda _: SimpleNamespace(
        time=server_wall_epoch,
        time_msc=server_wall_epoch * 1000,
        bid=1.1, ask=1.1001,
    )
    adapter = MT5DataAdapter(terminal=api, server_timezone="Etc/GMT-2", now=lambda: now)
    assert adapter.quote("EURUSD").time == now

    server_times = pd.date_range(end=now - pd.Timedelta(minutes=2), periods=4,
                                 freq="min", tz="UTC") + pd.Timedelta(hours=2)
    api.rates = pd.DataFrame({"time": [int(x.timestamp()) for x in server_times],
                              "open": [1.1]*4, "high": [1.2]*4,
                              "low": [1.0]*4, "close": [1.15]*4})
    candles = adapter.candles("EURUSD", "M1", 4)
    assert candles.time.iloc[-1] == now - pd.Timedelta(minutes=2)
    assert str(candles.time.dt.tz) == "UTC"


def test_naive_clock_is_rejected_instead_of_using_implicit_local_timezone():
    now = datetime(2026, 1, 1)
    adapter = MT5DataAdapter(terminal=FakeMT5(now.replace(tzinfo=timezone.utc)), now=lambda: now)
    with pytest.raises(MT5Error, match="timezone-aware"):
        adapter.quote("EURUSD")


def test_detects_terminal_drop_and_retries_initialize():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    api = FakeMT5(now)
    adapter = MT5DataAdapter(terminal=api, now=lambda: now)
    adapter.quote("EURUSD")
    api.online = False
    assert adapter.status()["connected"] is False
    api.online = True
    adapter.connect()
    assert api.initialized == 2


def test_resolves_a_unique_broker_suffix_and_rejects_ambiguous_names():
    api = FakeMT5(datetime(2026, 1, 1, tzinfo=timezone.utc))
    api.symbol_select = lambda symbol, enabled: symbol.endswith(".a")
    api.symbols_get = lambda: (SimpleNamespace(name="EURUSD.a"),)
    adapter = MT5DataAdapter(terminal=api)
    assert adapter.resolve_symbol("EURUSD") == "EURUSD.a"
    api.symbols_get = lambda: (SimpleNamespace(name="EURUSD.a"),
                               SimpleNamespace(name="EURUSD.pro"))
    with pytest.raises(MT5Error, match="ambiguous"):
        adapter.resolve_symbol("EURUSD")


def test_reads_only_completed_candles_and_rejects_invalid_ohlc():
    now = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    times = pd.date_range(end=now - pd.Timedelta(minutes=2), periods=4, freq="min", tz="UTC")
    rates = pd.DataFrame({"time": [int(x.timestamp()) for x in times], "open": [1.1]*4,
                          "high": [1.2]*4, "low": [1.0]*4, "close": [1.15]*4})
    api = FakeMT5(now, rates=rates)
    adapter = MT5DataAdapter(terminal=api, now=lambda: now)
    data = adapter.candles("EURUSD", "M1", 4)
    assert len(data) == 4
    assert str(data.time.dt.tz) == "UTC"
    api.rates = rates.assign(high=[1.0, 1.0, 1.0, 1.0])
    with pytest.raises(MT5Error, match="Invalid M1 OHLC"):
        adapter.candles("EURUSD", "M1", 4)


def test_rejects_stale_candles_and_unsupported_timeframe():
    now = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
    old = pd.date_range(end=now - pd.Timedelta(days=3), periods=4, freq="min", tz="UTC")
    rates = pd.DataFrame({"time": [int(x.timestamp()) for x in old], "open": [1.1]*4,
                          "high": [1.2]*4, "low": [1.0]*4, "close": [1.15]*4})
    adapter = MT5DataAdapter(terminal=FakeMT5(now, rates=rates), now=lambda: now)
    with pytest.raises(MT5Error, match="Stale M1 candles"):
        adapter.candles("EURUSD", "M1", 4)
    with pytest.raises(MT5Error, match="Unsupported timeframe"):
        adapter.candles("EURUSD", "H2", 4)


def test_rejects_future_candles_after_timezone_normalization():
    now = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
    future = pd.date_range(start=now + pd.Timedelta(minutes=1), periods=4,
                           freq="min", tz="UTC")
    rates = pd.DataFrame({"time": [int(x.timestamp()) for x in future],
                          "open": [1.1]*4, "high": [1.2]*4,
                          "low": [1.0]*4, "close": [1.15]*4})
    adapter = MT5DataAdapter(terminal=FakeMT5(now, rates=rates), now=lambda: now)
    with pytest.raises(MT5Error, match="Stale M1 candles"):
        adapter.candles("EURUSD", "M1", 4)
