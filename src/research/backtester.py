"""Sequential OHLC backtesting for precomputed candidate signals.

Input signals require ``decision_timestamp`` and ``direction`` columns, where
direction is ``long``/``short`` or +1/-1. The engine enters at the first candle
open strictly after the decision time plus ``execution_delay`` and exits at
the close of the first candle ending at or after entry plus ``holding_period``.
Timestamp convention must be supplied. For the project CSV, available evidence
supports Dukascopy BID M1 ``bar_open`` timestamps: values match the UTC-aware
parquet export after its timezone marker was removed. The exact CSV export
invocation is unavailable. The data loader localizes those naive CSV clock
values to UTC; the backtester requires timezone-aware custom inputs and
normalizes them to UTC, rejecting naive custom inputs without source timezone
context.

``transaction_cost_pips`` is a fixed per-round-trip placeholder. It does not
model historical spread, commissions, slippage, or fill quality. Results are
research mechanics only, not evidence of profitability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import load_market_data, normalize_utc_timestamps, validate_ohlc_prices
from .execution_models import (
    BidAskExecutionModel,
    MarketDataMetadata,
    prepare_bid_ask_inputs,
)


TRADE_COLUMNS = [
    "signal_id",
    "source_row_index",
    "source_candle_timestamp",
    "direction",
    "decision_timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "gross_pnl_pips",
    "transaction_costs_pips",
    "net_pnl_pips",
]

REJECTION_REASONS = (
    "invalid_decision_timestamp",
    "invalid_direction",
    "unavailable_entry",
    "invalid_entry_price",
    "unavailable_exit",
    "invalid_exit_price",
    "gap_exceeds_tolerance",
    "overlap_skipped",
)

BID_ASK_REJECTION_REASONS = REJECTION_REASONS + (
    "unavailable_entry_quote",
    "unavailable_exit_quote",
    "invalid_entry_spread",
    "invalid_exit_spread",
)

BID_ASK_TRADE_COLUMNS = TRADE_COLUMNS + [
    "execution_model_id",
    "execution_model_version",
    "entry_price_side",
    "exit_price_side",
    "mid_price_pnl_pips",
    "spread_cost_pips",
    "commission_costs_pips",
    "slippage_costs_pips",
]


def _is_valid_price(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _direction(value: Any) -> tuple[str, int] | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "long":
            return "long", 1
        if normalized == "short":
            return "short", -1
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    if not np.isfinite(value):
        return None
    if value == 1:
        return "long", 1
    if value == -1:
        return "short", -1
    return None


def _summarize(trades: pd.DataFrame) -> dict[str, Any]:
    """Summarize realized P&L; break-even trades count in win-rate denominator."""
    returns = trades["net_pnl_pips"].to_numpy(dtype=float)
    # Price subtraction can leave tiny floating-point residues around zero.
    break_even = np.isclose(returns, 0.0, rtol=0.0, atol=1e-9)
    returns = np.where(break_even, 0.0, returns)
    count = len(returns)
    winners = returns > 0
    losers = returns < 0
    positive_total = float(returns[winners].sum())
    negative_total = float(abs(returns[losers].sum()))

    # Profit factor is net winning pips / absolute net losing pips. It is
    # infinite when there are gains and no losses; undefined for no trades or
    # an all-break-even sample.
    if negative_total:
        profit_factor: float | None = positive_total / negative_total
    elif positive_total:
        profit_factor = float("inf")
    else:
        profit_factor = None

    if count:
        # All trades closed at one timestamp are one portfolio event. Aggregate
        # them first so arbitrary ordering among simultaneous exits cannot
        # create a different interim equity peak or drawdown.
        by_exit_time = trades.assign(_summary_return=returns).groupby(
            "exit_timestamp", sort=True
        )["_summary_return"].sum().to_numpy(dtype=float)
        equity = np.r_[0.0, np.cumsum(by_exit_time)]
        running_peak = np.maximum.accumulate(equity)
        max_drawdown = float((running_peak - equity).max())
        win_rate: float | None = float(winners.mean())
        average: float | None = float(returns.mean())
        total = float(returns.sum())
    else:
        max_drawdown = 0.0
        win_rate = None
        average = None
        total = 0.0

    return {
        "number_of_trades": count,
        "win_rate": win_rate,
        "break_even_trades": int(break_even.sum()),
        "total_net_pips": total,
        "average_net_pips": average,
        "profit_factor": profit_factor,
        "max_drawdown_pips": max_drawdown,
    }


def run_backtest(
    signals: pd.DataFrame,
    market_data: pd.DataFrame | None = None,
    *,
    holding_period: str | pd.Timedelta,
    timestamp_convention: str,
    execution_delay: str | pd.Timedelta = "0s",
    allow_overlapping: bool = False,
    transaction_cost_pips: float = 0.0,
    pip_size: float = 0.0001,
    bar_duration: str | pd.Timedelta = "1min",
    gap_tolerance: str | pd.Timedelta = "90s",
    execution_model: BidAskExecutionModel | None = None,
    ask_data: pd.DataFrame | None = None,
    bid_metadata: MarketDataMetadata | None = None,
    ask_metadata: MarketDataMetadata | None = None,
) -> dict[str, Any]:
    """Run candidate signals sequentially and return trades, summary, rejects.

    Required signal columns are ``decision_timestamp`` and ``direction``.
    ``holding_period`` is elapsed time from actual entry open to intended exit.
    Set ``allow_overlapping=True`` to retain entries whose position is still
    open; otherwise a candidate is skipped unless its entry is strictly after
    the previous accepted trade's exit timestamp.

    Entry uses the first candle open with timestamp strictly later than
    decision time plus ``execution_delay``. Exit uses the first candle close
    whose bar end is at or after the holding-period target. Gaps greater than
    ``gap_tolerance`` between the decision candle and exit candle are rejected.

    Summary conventions: a net break-even trade is not a win or loss but stays
    in the win-rate denominator. Profit factor uses winning net pips divided by
    absolute losing net pips; it is infinity if there are wins and no losses,
    and ``None`` when there are no wins/losses (including no trades). Drawdown
    is the largest peak-to-trough decline in cumulative net pips, ordered by
    exit time. A fixed ``transaction_cost_pips`` value is deducted once per
    round trip and is only a placeholder, not historical spread or slippage.

    Returns a dict with ``trades`` (DataFrame), ``summary`` (dict), and
    ``rejected`` (reason counts).
    """
    if timestamp_convention not in {"bar_open", "bar_close"}:
        raise ValueError("timestamp_convention must be 'bar_open' or 'bar_close'")
    holding = pd.Timedelta(holding_period)
    delay = pd.Timedelta(execution_delay)
    bar = pd.Timedelta(bar_duration)
    gap_limit = pd.Timedelta(gap_tolerance)
    if holding <= pd.Timedelta(0) or bar <= pd.Timedelta(0):
        raise ValueError("holding_period and bar_duration must be positive")
    if delay < pd.Timedelta(0) or gap_limit < pd.Timedelta(0):
        raise ValueError("execution_delay and gap_tolerance cannot be negative")
    if not np.isfinite(transaction_cost_pips) or transaction_cost_pips < 0:
        raise ValueError("transaction_cost_pips must be finite and nonnegative")
    if not np.isfinite(pip_size) or pip_size <= 0:
        raise ValueError("pip_size must be finite and positive")
    if not {"decision_timestamp", "direction"}.issubset(signals.columns):
        raise ValueError("signals must contain decision_timestamp and direction")

    if execution_model is None:
        if ask_data is not None or bid_metadata is not None or ask_metadata is not None:
            raise ValueError("BID/ASK inputs require an explicit side-aware execution_model")
        ask_market = None
        pair_report = None
        reasons = REJECTION_REASONS
        market = load_market_data() if market_data is None else market_data.copy()
    else:
        if not isinstance(execution_model, BidAskExecutionModel):
            raise TypeError("execution_model must be a BidAskExecutionModel")
        if transaction_cost_pips != 0.0:
            raise ValueError("fixed transaction_cost_pips must be zero in BID/ASK mode")
        if ask_data is None or bid_metadata is None or ask_metadata is None:
            raise ValueError("BID/ASK mode requires ASK data and metadata for both sides")
        bid_source = load_market_data() if market_data is None else market_data.copy()
        market, ask_market, pair_report = prepare_bid_ask_inputs(
            bid_source, ask_data, bid_metadata=bid_metadata, ask_metadata=ask_metadata,
            timestamp_convention=timestamp_convention, bar_duration=bar_duration,
        )
        reasons = BID_ASK_REJECTION_REASONS
    source_load_report = dict(market.attrs.get("load_report", {}))
    market_input_rows = len(market)
    if not {"time", "open", "high", "low", "close"}.issubset(market.columns):
        raise ValueError("market_data must contain time and OHLC columns")
    market["time"] = normalize_utc_timestamps(market["time"])
    invalid_timestamp = market["time"].isna()
    dropped_invalid_timestamps = int(invalid_timestamp.sum())
    market = market.loc[~invalid_timestamp].copy()
    market = validate_ohlc_prices(market)
    duplicate_timestamps = int(market["time"].duplicated(keep="last").sum())
    market = (market.sort_values("time")
              .drop_duplicates("time", keep="last")
              .reset_index(drop=True))
    if market.empty:
        raise ValueError("market_data has no valid timestamps")

    times = pd.DatetimeIndex(market["time"]).to_numpy(dtype="datetime64[ns]")
    if timestamp_convention == "bar_open":
        candle_opens = times
        candle_ends = times + bar.to_timedelta64()
    else:
        candle_opens = times - bar.to_timedelta64()
        candle_ends = times
    opens = market["open"].to_numpy(dtype=float)
    closes = market["close"].to_numpy(dtype=float)

    candidates = signals.copy()
    candidates["_original_order"] = np.arange(len(candidates))
    candidates["_decision"] = normalize_utc_timestamps(
        candidates["decision_timestamp"]
    )
    if "source_candle_timestamp" in candidates:
        candidates["_source_candle_timestamp"] = normalize_utc_timestamps(
            candidates["source_candle_timestamp"]
        )
    else:
        candidates["_source_candle_timestamp"] = pd.Series(
            pd.NaT, index=candidates.index, dtype="datetime64[ns, UTC]"
        )
    candidates = candidates.sort_values(
        ["_decision", "_original_order"], na_position="last", kind="stable"
    )

    rejected = {reason: 0 for reason in reasons}
    records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    last_exit_time: pd.Timestamp | None = None
    gap_value = gap_limit.to_timedelta64()
    if execution_model is not None:
        # This is an exact UTC timestamp join, never a row-position pairing.
        quotes = market[["time"]].merge(
            ask_market[["time", "open", "close"]].rename(columns={
                "open": "ask_open", "close": "ask_close",
            }), on="time", how="left", validate="one_to_one",
        )
        ask_opens = quotes["ask_open"].to_numpy(dtype=float)
        ask_closes = quotes["ask_close"].to_numpy(dtype=float)
    else:
        ask_opens = ask_closes = None

    def reject(signal: pd.Series, reason: str) -> None:
        rejected[reason] += 1
        rejected_records.append({
            "signal_id": signal.get("signal_id", pd.NA),
            "source_row_index": signal.get("source_row_index", pd.NA),
            "source_candle_timestamp": signal.get("_source_candle_timestamp", pd.NaT),
            "decision_timestamp": signal.get("_decision", pd.NaT),
            "direction": signal.get("direction", pd.NA),
            "rejection_reason": reason,
        })

    for _, signal in candidates.iterrows():
        decision = signal["_decision"]
        if pd.isna(decision):
            reject(signal, "invalid_decision_timestamp")
            continue
        direction = _direction(signal["direction"])
        if direction is None:
            reject(signal, "invalid_direction")
            continue
        direction_name, sign = direction

        entry_idx = int(np.searchsorted(
            candle_opens,
            (decision + delay).to_datetime64(),
            side="right",
        ))
        if entry_idx >= len(times):
            reject(signal, "unavailable_entry")
            continue
        if not _is_valid_price(opens[entry_idx]):
            reject(signal, "invalid_entry_price")
            continue
        if execution_model is not None:
            ask_entry = ask_opens[entry_idx]
            if not _is_valid_price(ask_entry):
                reject(signal, "unavailable_entry_quote")
                continue
            if ask_entry < opens[entry_idx]:
                reject(signal, "invalid_entry_spread")
                continue
        entry_time = pd.to_datetime(candle_opens[entry_idx], utc=True)

        exit_target = entry_time + holding
        exit_idx = int(np.searchsorted(candle_ends, exit_target.to_datetime64(), side="left"))
        if exit_idx >= len(times) or exit_idx < entry_idx:
            reject(signal, "unavailable_exit")
            continue
        if not _is_valid_price(closes[exit_idx]):
            reject(signal, "invalid_exit_price")
            continue
        if execution_model is not None:
            ask_exit = ask_closes[exit_idx]
            if not _is_valid_price(ask_exit):
                reject(signal, "unavailable_exit_quote")
                continue
            if ask_exit < closes[exit_idx]:
                reject(signal, "invalid_exit_spread")
                continue

        decision_bar_idx = int(np.searchsorted(candle_ends, decision.to_datetime64(), side="right") - 1)
        if decision_bar_idx < 0:
            reject(signal, "unavailable_entry")
            continue
        if exit_idx > decision_bar_idx and (
            np.diff(times[decision_bar_idx:exit_idx + 1]) > gap_value
        ).any():
            reject(signal, "gap_exceeds_tolerance")
            continue

        exit_time = pd.to_datetime(candle_ends[exit_idx], utc=True)
        if not allow_overlapping and last_exit_time is not None and entry_time <= last_exit_time:
            reject(signal, "overlap_skipped")
            continue

        if execution_model is None:
            entry_price = float(opens[entry_idx])
            exit_price = float(closes[exit_idx])
            gross = sign * (exit_price - entry_price) / pip_size
            net = gross - transaction_cost_pips
            record = {
            "signal_id": signal.get("signal_id", pd.NA),
            "source_row_index": signal.get("source_row_index", pd.NA),
            "source_candle_timestamp": signal["_source_candle_timestamp"],
            "direction": direction_name,
            "decision_timestamp": pd.Timestamp(decision),
            "entry_timestamp": entry_time,
            "exit_timestamp": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl_pips": float(gross),
            "transaction_costs_pips": float(transaction_cost_pips),
            "net_pnl_pips": float(net),
            }
        else:
            bid_entry = float(opens[entry_idx])
            ask_entry = float(ask_opens[entry_idx])
            bid_exit = float(closes[exit_idx])
            ask_exit = float(ask_closes[exit_idx])
            entry_price = ask_entry if direction_name == "long" else bid_entry
            exit_price = bid_exit if direction_name == "long" else ask_exit
            gross = sign * (exit_price - entry_price) / pip_size
            mid_entry = (bid_entry + ask_entry) / 2.0
            mid_exit = (bid_exit + ask_exit) / 2.0
            mid_gross = sign * (mid_exit - mid_entry) / pip_size
            spread_cost = ((ask_entry - bid_entry) + (ask_exit - bid_exit)) / (2.0 * pip_size)
            commission = float(execution_model.commission_pips_round_trip)
            slippage = float(execution_model.slippage_pips_round_trip)
            if not np.isclose(gross, mid_gross - spread_cost, rtol=0.0, atol=1e-8):
                raise AssertionError("side-aware gross P&L and spread accounting do not reconcile")
            explicit_costs = commission + slippage
            net = gross - explicit_costs
            record = {
                "signal_id": signal.get("signal_id", pd.NA),
                "source_row_index": signal.get("source_row_index", pd.NA),
                "source_candle_timestamp": signal["_source_candle_timestamp"],
                "direction": direction_name,
                "decision_timestamp": pd.Timestamp(decision),
                "entry_timestamp": entry_time,
                "exit_timestamp": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl_pips": float(gross),
                "transaction_costs_pips": float(explicit_costs),
                "net_pnl_pips": float(net),
                "execution_model_id": execution_model.identifier,
                "execution_model_version": execution_model.version,
                "entry_price_side": "ASK" if direction_name == "long" else "BID",
                "exit_price_side": "BID" if direction_name == "long" else "ASK",
                "mid_price_pnl_pips": float(mid_gross),
                "spread_cost_pips": float(spread_cost),
                "commission_costs_pips": commission,
                "slippage_costs_pips": slippage,
            }
        records.append(record)
        if not allow_overlapping:
            last_exit_time = exit_time

    trade_columns = TRADE_COLUMNS if execution_model is None else BID_ASK_TRADE_COLUMNS
    trades = pd.DataFrame(records, columns=trade_columns)
    if not trades.empty:
        trades = trades.sort_values(
            ["exit_timestamp", "entry_timestamp", "decision_timestamp"],
            kind="stable",
        ).reset_index(drop=True)
    result = {
        "trades": trades,
        "summary": _summarize(trades),
        "rejected": rejected,
        "rejected_signals": pd.DataFrame(rejected_records, columns=[
            "signal_id", "source_row_index", "source_candle_timestamp",
            "decision_timestamp", "direction", "rejection_reason",
        ]),
        "market_data_report": {
            "input_rows": market_input_rows,
            "dropped_invalid_timestamps": dropped_invalid_timestamps,
            "dropped_duplicate_timestamps": duplicate_timestamps,
            "rows_used": len(market),
            "source_load_report": source_load_report,
        },
    }
    if execution_model is not None:
        result["execution_model"] = {
            "identifier": execution_model.identifier,
            "version": execution_model.version,
            "alignment_policy": execution_model.alignment_policy,
            "alignment_report": pair_report,
            "commission_pips_round_trip": execution_model.commission_pips_round_trip,
            "slippage_pips_round_trip": execution_model.slippage_pips_round_trip,
            "commission_and_slippage_are_assumptions": True,
            "legacy_fixed_round_trip_cost_applied": False,
        }
        result["market_data_report"]["bid_ask_alignment"] = pair_report
    return result
