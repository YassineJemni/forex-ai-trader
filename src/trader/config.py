from __future__ import annotations

import os
import math
from dataclasses import dataclass


DEFAULT_PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}
# Read-only verification on the connected MetaQuotes-Demo terminal measured
# server-wall timestamps at UTC+2. Operators can override this after rechecking
# the broker clock, especially around daylight-saving changes.
DEFAULT_MT5_SERVER_TIMEZONE = "Etc/GMT-2"


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _optional_nonnegative_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative when supplied")
    return value


@dataclass(frozen=True)
class AppConfig:
    pairs: tuple[str, ...] = DEFAULT_PAIRS
    timeframes: tuple[str, ...] = ("M15", "H1", "H4")
    account_balance: float = 10_000.0
    account_currency: str = "USD"
    risk_percent: float = 0.5
    max_open_risk_percent: float = 2.0
    max_quote_age_seconds: float = 90.0
    bars_per_scan: int = 500
    mt5_terminal_path: str | None = None
    mt5_server_timezone: str = DEFAULT_MT5_SERVER_TIMEZONE
    research_model_path: str | None = None
    commission_pips_round_trip: float | None = None
    slippage_pips_round_trip: float | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        pairs = tuple(dict.fromkeys(x.strip().upper() for x in os.getenv("FOREX_PAIRS", ",".join(DEFAULT_PAIRS)).split(",") if x.strip()))
        timeframes = tuple(dict.fromkeys(x.strip().upper() for x in os.getenv("FOREX_TIMEFRAMES", "M15,H1,H4").split(",") if x.strip()))
        if not pairs:
            raise ValueError("FOREX_PAIRS must contain at least one symbol")
        unsupported = set(timeframes) - set(TIMEFRAME_MINUTES)
        if unsupported:
            raise ValueError(f"Unsupported timeframes: {sorted(unsupported)}")
        if not timeframes:
            raise ValueError("FOREX_TIMEFRAMES must contain at least one timeframe")
        bars = int(os.getenv("FOREX_BARS_PER_SCAN", "500"))
        if bars < 260:
            raise ValueError("FOREX_BARS_PER_SCAN must be at least 260 for feature warm-up")
        balance = _positive_float("FOREX_ACCOUNT_BALANCE", 10_000.0)
        risk = _positive_float("FOREX_RISK_PERCENT", 0.5)
        max_risk = _positive_float("FOREX_MAX_OPEN_RISK_PERCENT", 2.0)
        if risk > max_risk or max_risk > 100:
            raise ValueError("risk settings must satisfy per-trade risk <= max open risk <= 100")
        return cls(
            pairs=pairs, timeframes=timeframes,
            account_balance=balance,
            account_currency=os.getenv("FOREX_ACCOUNT_CURRENCY", "USD").upper(),
            risk_percent=risk,
            max_open_risk_percent=max_risk,
            max_quote_age_seconds=_positive_float("FOREX_MAX_QUOTE_AGE_SECONDS", 90.0),
            bars_per_scan=bars,
            mt5_terminal_path=os.getenv("MT5_TERMINAL_PATH") or None,
            mt5_server_timezone=os.getenv("MT5_SERVER_TIMEZONE", DEFAULT_MT5_SERVER_TIMEZONE).strip() or DEFAULT_MT5_SERVER_TIMEZONE,
            research_model_path=os.getenv("FOREX_RESEARCH_MODEL_PATH") or None,
            commission_pips_round_trip=_optional_nonnegative_float("FOREX_COMMISSION_PIPS_RT"),
            slippage_pips_round_trip=_optional_nonnegative_float("FOREX_SLIPPAGE_PIPS_RT"),
        )
