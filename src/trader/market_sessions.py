"""UTC-only Forex session labels with editable, explicit clock windows.

Windows are half-open UTC ranges. The four requested sessions are present,
with rollover separately tagged. The London/New York overlap is an additional
tag, so it does not replace the primary New York session.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Mapping


DEFAULT_SESSION_WINDOWS = {
    "Asia": ("00:00", "07:00"),
    "London": ("07:00", "13:00"),
    "NewYork": ("13:00", "21:00"),
    "LondonNewYorkOverlap": ("13:00", "17:00"),
    "Rollover": ("21:00", "00:00"),
}


def _clock(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":", maxsplit=1))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        if (hour, minute) != (24, 0):
            raise ValueError(f"Invalid UTC session clock: {value}")
    return time(hour=0) if (hour, minute) == (24, 0) else time(hour, minute)


def classify_utc_sessions(value: datetime, windows: Mapping[str, tuple[str, str]] | None = None) -> list[str]:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Session classification requires a timezone-aware timestamp")
    utc = value.astimezone(timezone.utc).time().replace(tzinfo=None)
    result = []
    for name, (start_text, end_text) in (windows or DEFAULT_SESSION_WINDOWS).items():
        start, end = _clock(start_text), _clock(end_text)
        active = start <= utc < end if start < end else (utc >= start or utc < end)
        if active:
            result.append(name)
    return result


def parse_session_windows(items: list[str] | None) -> dict[str, tuple[str, str]]:
    if not items:
        return dict(DEFAULT_SESSION_WINDOWS)
    output = {}
    for item in items:
        try:
            name, interval = item.split("=", maxsplit=1)
            start, end = interval.split("-", maxsplit=1)
            _clock(start)
            _clock(end)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Expected NAME=HH:MM-HH:MM, got {item!r}") from exc
        output[name] = (start, end)
    return output
