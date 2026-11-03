from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from .config import TIMEFRAME_MINUTES

from src.research.data import validate_ohlc_prices


class MT5Error(RuntimeError):
    """A safe, user-displayable MetaTrader 5 data error."""


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    time: datetime
    raw_time_seconds: int | None = None
    raw_time_milliseconds: int | None = None

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class MT5DataAdapter:
    """Lazy, read-only adapter around the MetaTrader5 Python package.

    The adapter intentionally exposes only ticks and completed candle history.
    It has no order_send wrapper or trading API surface.
    """

    def __init__(self, *, terminal: Any | None = None, terminal_path: str | None = None,
                 max_quote_age_seconds: float = 90.0,
                 server_timezone: str = "UTC",
                 now: Callable[[], datetime] | None = None):
        self._terminal = terminal
        self.terminal_path = terminal_path
        self.max_quote_age_seconds = max_quote_age_seconds
        try:
            ZoneInfo(server_timezone)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ValueError(f"Unknown MT5 server timezone: {server_timezone}") from exc
        self.server_timezone = server_timezone
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._connected = False
        self._last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def connect(self) -> None:
        if self._connected:
            checker = getattr(self._terminal, "terminal_info", None)
            try:
                alive = not callable(checker) or checker() is not None
            except Exception:
                alive = False
            if alive:
                return
            self._connected = False
            try:
                self._terminal.shutdown()
            except Exception:
                pass
        try:
            if self._terminal is None:
                import MetaTrader5 as mt5  # type: ignore[import-not-found]
                self._terminal = mt5
            kwargs = {"path": self.terminal_path} if self.terminal_path else {}
            initialized = self._terminal.initialize(**kwargs)
            if not initialized:
                error = self._terminal.last_error()
                raise MT5Error(f"MetaTrader 5 initialize failed: {error}")
            self._connected = True
            self._last_error = None
        except ImportError as exc:
            self._last_error = "MetaTrader5 Python package is not installed"
            raise MT5Error(self._last_error) from exc
        except MT5Error as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = f"MetaTrader 5 connection failed: {exc}"
            raise MT5Error(self._last_error) from exc

    def close(self) -> None:
        if self._terminal is not None and self._connected:
            self._terminal.shutdown()
        self._connected = False

    def status(self) -> dict[str, Any]:
        connected = self._connected
        if connected:
            checker = getattr(self._terminal, "terminal_info", None)
            if callable(checker):
                try:
                    connected = checker() is not None
                except Exception:
                    connected = False
                if not connected:
                    self._connected = False
                    self._last_error = "MetaTrader 5 terminal is unavailable; FETCH will retry the connection."
        return {"connected": connected, "read_only": True, "last_error": self._last_error}

    def raw_tick(self, symbol: str) -> Any:
        """Return one unmodified MT5 tick for read-only audit/observation use."""
        self.connect()
        if not self._terminal.symbol_select(symbol, True):
            self._last_error = f"MT5 symbol is unavailable: {symbol}"
            raise MT5Error(self._last_error)
        tick = self._terminal.symbol_info_tick(symbol)
        if tick is None:
            self._last_error = f"No current quote for {symbol}: {self._terminal.last_error()}"
            raise MT5Error(self._last_error)
        return tick

    def quote_from_tick(self, symbol: str, tick: Any) -> Quote:
        """Normalize and validate a previously read raw tick exactly once."""
        seconds = int(getattr(tick, "time", 0) or 0)
        millis = int(getattr(tick, "time_msc", 0) or 0)
        when = self.normalize_tick_timestamp(tick, symbol)
        age = (self._utc_now() - when).total_seconds()
        if age < -5:
            raise MT5Error(f"MT5 quote timestamp is in the future for {symbol}")
        if age > self.max_quote_age_seconds:
            raise MT5Error(f"Stale MT5 quote for {symbol} ({age:.0f}s old)")
        bid, ask = float(tick.bid), float(tick.ask)
        if not (0 < bid <= ask):
            raise MT5Error(f"Invalid or crossed quote for {symbol}")
        return Quote(symbol=symbol, bid=bid, ask=ask, time=when,
                     raw_time_seconds=seconds, raw_time_milliseconds=millis or None)

    def quote(self, symbol: str) -> Quote:
        return self.quote_from_tick(symbol, self.raw_tick(symbol))

    def normalize_tick_timestamp(self, tick: Any, symbol: str) -> datetime:
        """Normalize raw tick time while preserving its original epoch fields."""
        seconds = int(getattr(tick, "time", 0) or 0)
        millis = int(getattr(tick, "time_msc", 0) or 0)
        if millis:
            when = self._mt5_server_epoch_to_utc(millis, "ms", symbol)
            if seconds:
                coarse = self._mt5_server_epoch_to_utc(seconds, "s", symbol)
                if abs((when - coarse).total_seconds()) > 1.5:
                    raise MT5Error(f"Inconsistent MT5 quote timestamps for {symbol}")
            return when
        if seconds:
            return self._mt5_server_epoch_to_utc(seconds, "s", symbol)
        raise MT5Error(f"MT5 quote has no valid timestamp for {symbol}")

    def candles(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        self.connect()
        constants = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5",
                     "M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1",
                     "H4": "TIMEFRAME_H4"}
        if timeframe not in constants:
            raise MT5Error(f"Unsupported timeframe: {timeframe}")
        if count < 2:
            raise ValueError("count must be at least 2")
        if not self._terminal.symbol_select(symbol, True):
            self._last_error = f"MT5 symbol is unavailable: {symbol}"
            raise MT5Error(self._last_error)
        tf = getattr(self._terminal, constants[timeframe])
        # start_pos=1 excludes the currently forming candle.
        rates = self._terminal.copy_rates_from_pos(symbol, tf, 1, count)
        if rates is None or len(rates) < 2:
            self._last_error = f"Insufficient {timeframe} candles for {symbol}: {self._terminal.last_error()}"
            raise MT5Error(self._last_error)
        data = pd.DataFrame(rates)
        if "time" not in data:
            raise MT5Error(f"MT5 candle response has no timestamps for {symbol}")
        try:
            data["time"] = self._mt5_candle_times_to_utc(data["time"])
        except (ValueError, TypeError) as exc:
            raise MT5Error(f"Invalid {timeframe} timestamps for {symbol}: {exc}") from exc
        if data["time"].duplicated().any():
            raise MT5Error(f"Duplicate {timeframe} candle timestamps for {symbol}")
        data = data.sort_values("time", kind="stable").reset_index(drop=True)
        try:
            data = validate_ohlc_prices(data)
        except ValueError as exc:
            raise MT5Error(f"Invalid {timeframe} OHLC data for {symbol}: {exc}") from exc
        latest = data["time"].iloc[-1].to_pydatetime()
        age = (self._utc_now() - latest).total_seconds()
        allowed_age = TIMEFRAME_MINUTES[timeframe] * 120 + self.max_quote_age_seconds
        if age < -5 or age > allowed_age:
            raise MT5Error(f"Stale {timeframe} candles for {symbol} ({age:.0f}s since latest completed bar)")
        return data

    def history_chunk(self, symbol: str, timeframe: str, start_pos: int,
                      count: int) -> pd.DataFrame:
        """Fetch raw completed history for read-only coverage audits.

        Unlike ``candles``, this preserves duplicate rows and does not reject
        stale, sparse, or OHLC-invalid responses; callers need those rows to
        report data quality. MT5 position zero is the forming bar, so callers
        must start at one or greater.
        """
        self.connect()
        constants = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5",
                     "M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1",
                     "H4": "TIMEFRAME_H4"}
        if timeframe not in constants:
            raise MT5Error(f"Unsupported timeframe: {timeframe}")
        if start_pos < 1 or count < 1:
            raise ValueError("start_pos must be >= 1 and count must be >= 1")
        if not self._terminal.symbol_select(symbol, True):
            self._last_error = f"MT5 symbol is unavailable: {symbol}"
            raise MT5Error(self._last_error)
        rates = self._terminal.copy_rates_from_pos(
            symbol, getattr(self._terminal, constants[timeframe]), start_pos, count
        )
        if rates is None:
            self._last_error = f"MT5 history API error: {self._terminal.last_error()}"
            raise MT5Error(self._last_error)
        self._last_error = None
        return pd.DataFrame(rates)

    def terminal_info(self) -> Any | None:
        """Return terminal metadata when available (read-only)."""
        self.connect()
        getter = getattr(self._terminal, "terminal_info", None)
        return getter() if callable(getter) else None

    def symbol_info(self, symbol: str) -> Any | None:
        """Read a symbol specification without changing account state."""
        self.connect()
        getter = getattr(self._terminal, "symbol_info", None)
        if not callable(getter):
            raise MT5Error("MT5 symbol metadata API is unavailable")
        value = getter(symbol)
        if value is None:
            self._last_error = f"No MT5 symbol metadata for {symbol}: {self._terminal.last_error()}"
            raise MT5Error(self._last_error)
        return value

    def account_info(self) -> Any | None:
        """Read account metadata only; this adapter exposes no account mutations."""
        self.connect()
        getter = getattr(self._terminal, "account_info", None)
        return getter() if callable(getter) else None

    def deal_history(self, start_utc: datetime, end_utc: datetime) -> list[Any]:
        """Read bounded deal history for fee diagnostics; never changes the account."""
        if start_utc.tzinfo is None or start_utc.utcoffset() is None or end_utc.tzinfo is None or end_utc.utcoffset() is None:
            raise ValueError("deal history bounds must be timezone-aware")
        if start_utc >= end_utc:
            raise ValueError("deal history start must precede end")
        self.connect()
        getter = getattr(self._terminal, "history_deals_get", None)
        if not callable(getter):
            raise MT5Error("MT5 deal history API is unavailable")
        deals = getter(start_utc.astimezone(timezone.utc), end_utc.astimezone(timezone.utc))
        if deals is None:
            self._last_error = f"Could not read bounded MT5 deal history: {self._terminal.last_error()}"
            raise MT5Error(self._last_error)
        return list(deals)

    def _utc_now(self) -> datetime:
        now = self._now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise MT5Error("Clock must return a timezone-aware datetime")
        return now.astimezone(timezone.utc)

    def _mt5_server_epoch_to_utc(self, value: int, unit: str, symbol: str) -> datetime:
        # MT5 normally documents these epochs as UTC. Some server feeds expose
        # broker wall time in the epoch fields; reinterpret that wall clock only
        # when the operator explicitly configures MT5_SERVER_TIMEZONE.
        wall = pd.Timestamp(pd.to_datetime(value, unit=unit, utc=True)).tz_localize(None)
        localized = wall.tz_localize(self.server_timezone, ambiguous="NaT", nonexistent="NaT")
        if pd.isna(localized):
            raise MT5Error(f"Ambiguous/nonexistent MT5 server timestamp for {symbol}")
        return localized.tz_convert("UTC").to_pydatetime()

    def _mt5_candle_times_to_utc(self, values: pd.Series) -> pd.Series:
        # MT5 rates use the same server-wall-clock epoch convention as ticks
        # for this configured terminal. Keep the conversion sequence identical.
        server_wall = pd.to_datetime(values, unit="s", utc=True).dt.tz_localize(None)
        localized = server_wall.dt.tz_localize(
            self.server_timezone, ambiguous="NaT", nonexistent="NaT"
        )
        if localized.isna().any():
            raise ValueError("timestamps include ambiguous or nonexistent server-local times")
        return localized.dt.tz_convert("UTC")

    def resolve_symbol(self, requested: str) -> str:
        """Resolve a canonical six-letter pair to one broker symbol safely."""
        self.connect()
        if self._terminal.symbol_select(requested, True):
            return requested
        symbols_get = getattr(self._terminal, "symbols_get", None)
        if not callable(symbols_get):
            raise MT5Error(f"MT5 symbol is unavailable: {requested}")
        canonical = "".join(ch for ch in requested.upper() if ch.isalpha())[:6]
        matches = []
        for info in symbols_get() or ():
            name = getattr(info, "name", "")
            compact = "".join(ch for ch in name.upper() if ch.isalpha())
            if compact == canonical or compact.startswith(canonical):
                matches.append(name)
        if len(matches) != 1:
            reason = "no match" if not matches else f"ambiguous matches: {matches}"
            raise MT5Error(f"Cannot resolve broker symbol {requested}: {reason}")
        if not self._terminal.symbol_select(matches[0], True):
            raise MT5Error(f"MT5 broker symbol is unavailable: {matches[0]}")
        return matches[0]

    def open_positions(self) -> list[Any]:
        """Read current account positions; never modifies them."""
        self.connect()
        positions = self._terminal.positions_get()
        if positions is None:
            self._last_error = f"Could not read MT5 open positions: {self._terminal.last_error()}"
            raise MT5Error(self._last_error)
        return list(positions)
