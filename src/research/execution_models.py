"""Versioned execution assumptions and strict BID/ASK input validation.

The legacy fixed-cost model remains the backtester's default. The side-aware
model is opt-in and uses exact UTC timestamp joins; it never fills missing
quotes by row position, forward-fill, or interpolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .data import normalize_utc_timestamps, validate_m1_cadence, validate_ohlc_prices


BID_ASK_EXECUTION_MODEL_ID = "bid_ask_side_aware"
BID_ASK_EXECUTION_MODEL_VERSION = "1.0.0"
BID_ASK_ALIGNMENT_POLICY = "exact_utc_timestamp_join_v1"


@dataclass(frozen=True)
class MarketDataMetadata:
    """Declared source facts required to pair two market sides safely."""

    side: str
    symbol: str
    timeframe: str
    timestamp_convention: str
    timezone: str
    provider: str

    def __post_init__(self) -> None:
        if self.side not in {"BID", "ASK"}:
            raise ValueError("market side must be BID or ASK")
        if not self.symbol.strip() or not self.provider.strip():
            raise ValueError("symbol and provider metadata are required")
        if self.timeframe != "M1":
            raise ValueError("market timeframe must be M1 for side-aware execution")
        if self.timestamp_convention not in {"bar_open", "bar_close"}:
            raise ValueError("timestamp_convention must be bar_open or bar_close")
        if self.timezone != "UTC":
            raise ValueError("side-aware execution requires UTC source timestamps")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class BidAskExecutionModel:
    """Side-aware fill model with explicit round-trip cost assumptions.

    Commission and slippage are fixed assumptions in pips per round trip. They
    are not estimated from OHLC data. BID/ASK spread is reflected by the chosen
    fill-side prices and is not deducted a second time.
    """

    commission_pips_round_trip: float
    slippage_pips_round_trip: float
    identifier: str = BID_ASK_EXECUTION_MODEL_ID
    version: str = BID_ASK_EXECUTION_MODEL_VERSION
    alignment_policy: str = BID_ASK_ALIGNMENT_POLICY

    def __post_init__(self) -> None:
        for name in ("commission_pips_round_trip", "slippage_pips_round_trip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if not pd.notna(value) or not float("-inf") < float(value) < float("inf") or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.identifier != BID_ASK_EXECUTION_MODEL_ID:
            raise ValueError("unsupported execution model identifier")
        if self.version != BID_ASK_EXECUTION_MODEL_VERSION:
            raise ValueError("unsupported execution model version")
        if self.alignment_policy != BID_ASK_ALIGNMENT_POLICY:
            raise ValueError("unsupported BID/ASK alignment policy")

    def identity_payload(
        self,
        *,
        bid_metadata: MarketDataMetadata,
        ask_metadata: MarketDataMetadata,
        bid_fingerprint: str,
        ask_fingerprint: str,
        alignment_report: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "version": self.version,
            "fill_rules": {
                "long_entry": "ASK open",
                "long_exit": "BID close",
                "short_entry": "BID open",
                "short_exit": "ASK close",
            },
            "data_side_provenance": {
                "bid": {**bid_metadata.to_dict(), "fingerprint_sha256": bid_fingerprint},
                "ask": {**ask_metadata.to_dict(), "fingerprint_sha256": ask_fingerprint},
            },
            "alignment_policy": self.alignment_policy,
            "alignment_report": dict(alignment_report),
            "cost_components": {
                "spread": "embedded in side-specific executable prices; not separately deducted",
                "commission_pips_round_trip": float(self.commission_pips_round_trip),
                "slippage_pips_round_trip": float(self.slippage_pips_round_trip),
                "commission_and_slippage_are_assumptions": True,
                "legacy_fixed_round_trip_cost_applied": False,
            },
        }


def _normalize_side_frame(data: pd.DataFrame, metadata: MarketDataMetadata) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{metadata.side} data must be a pandas DataFrame")
    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(data.columns):
        raise ValueError(f"{metadata.side} data must include {sorted(required)}")
    frame = data.copy()
    frame["time"] = normalize_utc_timestamps(frame["time"])
    if frame["time"].isna().any():
        raise ValueError(f"{metadata.side} timestamps must be valid and timezone-aware")
    if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise ValueError(f"{metadata.side} timestamps must be sorted and unique")
    frame = validate_ohlc_prices(frame)
    validate_m1_cadence(frame["time"])
    return frame.reset_index(drop=True)


def prepare_bid_ask_inputs(
    bid_data: pd.DataFrame,
    ask_data: pd.DataFrame,
    *,
    bid_metadata: MarketDataMetadata,
    ask_metadata: MarketDataMetadata,
    timestamp_convention: str,
    bar_duration: str | pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate source compatibility and report exact timestamp alignment.

    Returned frames stay separate. The backtester joins ASK values to BID bars
    by UTC timestamp for each required observation, allowing unmatched quotes
    to become explicit per-signal rejections.
    """
    if bid_metadata.side != "BID" or ask_metadata.side != "ASK":
        raise ValueError("metadata sides must be BID for bid_data and ASK for ask_data")
    for field in ("symbol", "timeframe", "timestamp_convention", "timezone", "provider"):
        left, right = getattr(bid_metadata, field), getattr(ask_metadata, field)
        if (left.casefold() if isinstance(left, str) else left) != (
            right.casefold() if isinstance(right, str) else right
        ):
            label = field.replace("_", " ")
            raise ValueError(f"BID and ASK {label} metadata must match")
    if bid_metadata.timestamp_convention != timestamp_convention:
        raise ValueError("market metadata timestamp convention does not match backtest convention")
    if pd.Timedelta(bar_duration) != pd.Timedelta("1min"):
        raise ValueError("side-aware execution currently supports one-minute bars only")

    bid = _normalize_side_frame(bid_data, bid_metadata)
    ask = _normalize_side_frame(ask_data, ask_metadata)
    # Exact timestamp-set comparison is intentional. We do not zip rows or
    # silently align shifted candles. Missing/extra observations are reported;
    # a missing required fill quote is rejected explicitly by the backtester.
    joined = bid[["time"]].merge(
        ask[["time"]], on="time", how="outer", validate="one_to_one", indicator=True,
    )
    bid_only = int(joined["_merge"].eq("left_only").sum())
    ask_only = int(joined["_merge"].eq("right_only").sum())
    return bid, ask, {
        "alignment_policy": BID_ASK_ALIGNMENT_POLICY,
        "bid_rows": int(len(bid)),
        "ask_rows": int(len(ask)),
        "matched_timestamps": int(joined["_merge"].eq("both").sum()),
        "bid_without_ask_timestamp": bid_only,
        "ask_without_bid_timestamp": ask_only,
        "timestamps_are_exactly_aligned": bid_only == 0 and ask_only == 0,
    }
