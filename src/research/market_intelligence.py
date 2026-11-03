"""Causal market context and paper-only coordination for the existing strategies.

All calculations are based on closed candles and the existing Mission 8 strategy
functions. Scores are ordering aids, never probabilities or performance claims.
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data_integrity import TIMEFRAME_MINUTES
from .multi_strategy_engine import STRATEGY_WARMUP, build_features, strategy_library


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_IDS = tuple(strategy.strategy_id for strategy in strategy_library())
PIP_SIZE = {"USDJPY": 0.01}
VALIDATION_SCENARIO = "spread_2_commission_1_slippage_0.5_pips_rt"
MIN_VALIDATION_TRADES = 25
MAX_CANDLE_AGE = {
    timeframe: pd.Timedelta(minutes=minutes, seconds=90)
    for timeframe, minutes in TIMEFRAME_MINUTES.items()
}
MAX_CORRELATION_FOR_INDEPENDENCE = 0.80


def load_validation_evidence(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load Mission 12 validation-only summaries; never reads its test split."""
    source = path or ROOT / "reports/strategy-research/mission12/mission12-20261103T185609Z.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = payload.get("strategy_diagnostics", {}).get("validation", [])
    return {str(row["strategy_id"]): dict(row) for row in rows if row.get("strategy_id")}


def load_pair_timeframe_validation(
    path: Path | None = None,
    *,
    stability_path: Path | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load only exact pair/timeframe Mission 9 validation rows."""
    source = path or ROOT / "reports/strategy-research/mission9/mission9_cost_research_20261103T101819Z.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if stability_path is None:
        stability_reports = sorted(
            (ROOT / "reports/strategy-research/mission13").glob("mission13_3_eligibility_*.json"))
        stability_path = stability_reports[-1] if stability_reports else None
    stability_status: dict[tuple[str, str, str], str] = {}
    if stability_path is not None and stability_path.is_file():
        stability = json.loads(stability_path.read_text(encoding="utf-8"))
        fingerprints = stability.get("source_fingerprints_sha256", {})
        expected_fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
        if fingerprints.get("mission9_cost_report_reference") == expected_fingerprint:
            for row in stability.get("cells", []):
                key = (str(row.get("strategy_id")), str(row.get("pair")), str(row.get("timeframe")))
                status = row.get("pair_timeframe_stability_status")
                if status in {"stable_pair_timeframe_validation",
                              "unstable_or_insufficient_validation"}:
                    stability_status[key] = status
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload.get("metrics", []):
        if row.get("split") == "validation" and row.get("scenario") == VALIDATION_SCENARIO:
            copied = dict(row)
            key = (str(row.get("strategy_id")), str(row.get("pair")), str(row.get("timeframe")))
            copied["pair_timeframe_stability_status"] = stability_status.get(
                key, "unstable_or_insufficient_validation")
            indexed[key] = copied
    for row in payload.get("market_regime", []):
        if row.get("split") == "validation" and row.get("scenario") == VALIDATION_SCENARIO:
            key = (str(row.get("strategy_id")), str(row.get("pair")), str(row.get("timeframe")))
            base = indexed.setdefault(key, {})
            base.setdefault("regime_metrics", {})[str(row.get("regime"))] = dict(row)
    return indexed


def load_validation_correlations(path: Path | None = None) -> dict[tuple[str, str], float]:
    """Load the Mission 11 validation-only strategy-return correlation matrix."""
    source = path or ROOT / "reports/strategy-research/mission11/mission11_portfolio_20261103T114226Z.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    matrix = payload.get("validation_strategy_correlation_matrix", [])
    result: dict[tuple[str, str], float] = {}
    for row in matrix:
        first = str(row.get("strategy_id", ""))
        for second, value in row.items():
            if not first or second in {"strategy_id", first}:
                continue
            correlation = _number(value)
            if correlation is not None and -1 <= correlation <= 1:
                result[tuple(sorted((first, str(second))))] = correlation
    return result


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def analyze_market(features: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Describe current structure using only confirmed/current-and-past bars."""
    if features.empty:
        raise ValueError("market feature frame is empty")
    row = features.iloc[-1]
    close, atr = _number(row.close), _number(row.atr14)
    if close is None or atr is None or atr <= 0:
        raise ValueError("market features are not warmed up")
    trend = "bullish" if row.ema21 > row.ema50 else "bearish" if row.ema21 < row.ema50 else "neutral"
    trend_strength = abs(float(row.ema21) - float(row.ema50)) / atr
    prior_hi20, prior_lo20 = _number(row.prior_high20), _number(row.prior_low20)
    prior_hi10, prior_lo10 = _number(row.prior_high10), _number(row.prior_low10)
    swing_hi = _number(row.prior_confirmed_swing_high)
    swing_lo = _number(row.prior_confirmed_swing_low)
    close_time = pd.Timestamp(row.time) + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    hour = close_time.hour
    session = ("Asia_UTC_00_06" if hour < 7 else "London_UTC_07_12" if hour < 13
               else "NewYork_UTC_13_20" if hour < 21 else "Rollover_UTC_21_23")
    def last_event(columns: tuple[str, ...]) -> str | None:
        events = []
        for column in columns:
            matches = np.flatnonzero(features[column].fillna(False).to_numpy(dtype=bool))
            if len(matches):
                events.append((int(matches[-1]), column))
        return max(events)[1] if events else None
    high_event = last_event(("higher_high", "lower_high"))
    low_event = last_event(("higher_low", "lower_low"))
    structure = ("higher_high_higher_low" if high_event == "higher_high" and low_event == "higher_low" else
                 "lower_high_lower_low" if high_event == "lower_high" and low_event == "lower_low" else
                 "mixed_or_unconfirmed")
    bos = "bullish" if swing_hi is not None and close > swing_hi else (
        "bearish" if swing_lo is not None and close < swing_lo else None)
    prior_trend = _number(row.structure_trend)
    choch = bool(bos and ((bos == "bullish" and prior_trend is not None and prior_trend < 0)
                          or (bos == "bearish" and prior_trend is not None and prior_trend > 0)))
    atr_median = _number(features.atr14.rolling(100, min_periods=30).median().iloc[-1])
    volatility_ratio = atr / atr_median if atr_median and atr_median > 0 else None
    momentum5 = _number(row.momentum5)
    rsi = _number(row.rsi14)
    channel_width_atr = ((prior_hi20 - prior_lo20) / atr
                         if prior_hi20 is not None and prior_lo20 is not None else None)
    breakout = "bullish" if prior_hi20 is not None and close > prior_hi20 else (
        "bearish" if prior_lo20 is not None and close < prior_lo20 else None)
    sweep = "bullish_reclaim" if prior_lo10 is not None and row.low < prior_lo10 and close > prior_lo10 else (
        "bearish_rejection" if prior_hi10 is not None and row.high > prior_hi10 and close < prior_hi10 else None)
    return {
        "timeframe": timeframe, "decision_timestamp": close_time.isoformat(), "close": close,
        "trend_direction": trend, "trend_strength_ema21_50_atr": trend_strength,
        "support_20": prior_lo20, "resistance_20": prior_hi20,
        "support_10": prior_lo10, "resistance_10": prior_hi10,
        "last_confirmed_swing_low": swing_lo, "last_confirmed_swing_high": swing_hi,
        "swing_sequence": structure,
        "most_recent_confirmed_swing_events": {"high": high_event, "low": low_event},
        "break_of_structure": bos,
        "potential_change_of_character": choch,
        "atr14": atr, "atr_vs_100_bar_median": volatility_ratio,
        "volatility_state": ("expanding" if volatility_ratio is not None and volatility_ratio > 1.5 else
                             "contracting" if volatility_ratio is not None and volatility_ratio < 0.7 else "typical_or_unknown"),
        "regime": str(row.regime), "rsi14": rsi, "momentum5": momentum5,
        "momentum_state": ("overbought_exhaustion_watch" if rsi is not None and rsi >= 70 else
                           "oversold_exhaustion_watch" if rsi is not None and rsi <= 30 else "not_extreme"),
        "channel_width_20_atr": channel_width_atr,
        "consolidation": bool(row.regime == "ranging" or (channel_width_atr is not None and channel_width_atr < 5.0)),
        "breakout_condition": breakout, "liquidity_sweep_hypothesis": sweep,
        "session_utc": session,
        "definitions": {"swing": "5-bar pivot confirmed two closed bars after its center",
                        "support_resistance": "prior 10/20-bar extrema excluding current candle",
                        "bos": "close beyond last confirmed swing level",
                        "sweep": "current high/low crosses prior 10-bar extreme and closes back inside",
                        "consolidation": "existing causal ranging regime or 20-bar channel narrower than 5 ATR"},
    }


def _utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _validate_market_input(
    timeframe: str,
    raw: pd.DataFrame,
    quote: Any,
    as_of: pd.Timestamp,
    max_quote_age_seconds: float,
) -> tuple[pd.DataFrame, str | None]:
    required = {"time", "open", "high", "low", "close"}
    if timeframe not in TIMEFRAME_MINUTES:
        return pd.DataFrame(), f"unsupported timeframe {timeframe}"
    if not required.issubset(raw.columns):
        return pd.DataFrame(), f"missing candle columns: {sorted(required - set(raw.columns))}"
    if raw.empty:
        return pd.DataFrame(), "no candles supplied"
    if quote is None:
        return pd.DataFrame(), "fresh quote unavailable"
    try:
        quote_time = _utc_timestamp(quote.time, "quote timestamp")
    except (AttributeError, TypeError, ValueError) as exc:
        return pd.DataFrame(), f"invalid quote timestamp: {exc}"
    bid, ask = _number(getattr(quote, "bid", None)), _number(getattr(quote, "ask", None))
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return pd.DataFrame(), "invalid bid/ask quote values"
    quote_age = (as_of - quote_time).total_seconds()
    if quote_age < 0:
        return pd.DataFrame(), "quote timestamp is after the shared as-of cutoff"
    if quote_age > max_quote_age_seconds:
        return pd.DataFrame(), f"stale quote ({quote_age:.0f}s old at shared as-of cutoff)"
    try:
        source_times = [pd.Timestamp(value) for value in raw["time"]]
        if any(value.tzinfo is None or value.utcoffset() is None for value in source_times):
            return pd.DataFrame(), "candle timestamps must be timezone-aware UTC values"
        opens = pd.DatetimeIndex([value.tz_convert("UTC") for value in source_times])
    except (TypeError, ValueError, OverflowError) as exc:
        return pd.DataFrame(), f"invalid candle timestamps: {exc}"
    if opens.has_duplicates or not opens.is_monotonic_increasing:
        return pd.DataFrame(), "duplicate or unordered candle timestamps"
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    if len(opens) > 1 and (opens.to_series().diff().dropna() < step).any():
        return pd.DataFrame(), "overlapping candle open/close intervals"
    closes = opens + step
    if (opens > as_of).any() or (closes > as_of).any():
        return pd.DataFrame(), "future or not-yet-closed candle at shared as-of cutoff"
    age = as_of - closes[-1]
    if age > MAX_CANDLE_AGE[timeframe]:
        return pd.DataFrame(), f"stale {timeframe} candles ({age.total_seconds():.0f}s since latest close)"
    frame = raw.copy().reset_index(drop=True)
    frame["time"] = opens
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (frame.high < frame[["open", "close", "low"]].max(axis=1)).any()
            or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any()):
        return pd.DataFrame(), "invalid OHLC candles"
    frame["candle_close_time"] = closes
    return frame, None


def _validation_evidence_for(
    strategy_id: str,
    pair: str,
    timeframe: str,
    regime: str,
    evidence: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    row = evidence.get((strategy_id, pair, timeframe))
    if row is None:
        return None, ["no validation evidence for this exact strategy/pair/timeframe"], None
    if (row.get("strategy_id") != strategy_id or row.get("pair") != pair
            or row.get("timeframe") != timeframe or row.get("split") != "validation"):
        return None, ["validation evidence scope or split does not match this market"], None
    if row.get("scenario") != VALIDATION_SCENARIO:
        return None, ["validation evidence does not use the declared cost scenario"], None
    reasons = []
    if row.get("pair_timeframe_stability_status") != "stable_pair_timeframe_validation":
        reasons.append("pair/timeframe-specific validation stability is not established")
    selected = row
    scope = "pair_timeframe"
    regime_rows = row.get("regime_metrics", {})
    if regime in regime_rows:
        selected = regime_rows[regime]
        scope = "pair_timeframe_regime"
        if (selected.get("strategy_id") != strategy_id or selected.get("pair") != pair
                or selected.get("timeframe") != timeframe or selected.get("split") != "validation"
                or selected.get("scenario") != VALIDATION_SCENARIO):
            reasons.append("regime evidence scope or split does not match this market")
    trades = int(selected.get("trades", 0) or 0)
    if trades < MIN_VALIDATION_TRADES:
        reasons.append(f"insufficient validation sample ({trades} < {MIN_VALIDATION_TRADES})")
    expectancy = _number(selected.get("expectancy_r", selected.get("expectancy_net_r")))
    if expectancy is None:
        reasons.append("validation expectancy is unavailable")
    elif expectancy <= 0:
        reasons.append("validation expectancy is nonpositive after declared costs")
    return (dict(selected) if not reasons else None), reasons, scope


def _independence_assessment(
    strategy_ids: list[str],
    correlations: Mapping[tuple[str, str], float],
) -> tuple[float, list[str], list[str], list[str]]:
    independent: list[str] = []
    correlated: list[str] = []
    unknown: list[str] = []
    established = [strategy_ids[0]] if strategy_ids else []
    for candidate in strategy_ids[1:]:
        values = [correlations.get(tuple(sorted((known, candidate)))) for known in established]
        if any(value is not None and abs(value) >= MAX_CORRELATION_FOR_INDEPENDENCE for value in values):
            correlated.append(candidate)
        elif not values or any(value is None for value in values):
            unknown.append(candidate)
        else:
            independent.append(candidate)
        established.append(candidate)
    return min(2.0, 0.75 * len(independent)), independent, correlated, unknown


def coordinate_market_set(
    candles: Mapping[tuple[str, str], pd.DataFrame],
    quotes: Mapping[str, Any],
    *,
    validation_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    pair_timeframe_evidence: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None,
    validation_correlations: Mapping[tuple[str, str], float] | None = None,
    as_of_utc: Any | None = None,
    max_quote_age_seconds: float = 90.0,
    commission_pips_round_trip: float | None = None,
    slippage_pips_round_trip: float | None = None,
    portfolio_exposure_clear: bool | None = None,
) -> dict[str, Any]:
    """Run every strategy on validated, time-aligned markets in paper mode.

    The legacy validation_evidence argument is retained for call compatibility
    but ignored because it has no pair/timeframe-scoped stability evidence.
    """
    strategies = strategy_library()
    pair_evidence = (pair_timeframe_evidence if pair_timeframe_evidence is not None
                     else load_pair_timeframe_validation())
    correlations = (validation_correlations if validation_correlations is not None
                    else load_validation_correlations())
    if not np.isfinite(max_quote_age_seconds) or max_quote_age_seconds <= 0:
        raise ValueError("max_quote_age_seconds must be positive")
    cutoff = as_of_utc if as_of_utc is not None else datetime.now(timezone.utc)
    as_of = _utc_timestamp(cutoff, "shared as-of cutoff")
    markets: dict[tuple[str, str], dict[str, Any]] = {}
    feature_timelines: dict[tuple[str, str], pd.DataFrame] = {}
    signals: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    evaluated = 0
    evaluated_by_market: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    market_rejections: list[dict[str, Any]] = []
    for (pair, timeframe), raw in sorted(candles.items()):
        frame, rejection = _validate_market_input(
            timeframe, raw, quotes.get(pair), as_of, max_quote_age_seconds)
        market_key = f"{pair}_{timeframe}"
        if rejection:
            market_rejections.append({"pair": pair, "timeframe": timeframe, "reason": rejection})
            continue
        if len(frame) < STRATEGY_WARMUP + 20:
            market_rejections.append({"pair": pair, "timeframe": timeframe,
                                      "reason": "insufficient candle history for strategy warm-up"})
            continue
        try:
            features = build_features(frame, timeframe)
            market = analyze_market(features, timeframe)
            quote = quotes[pair]
            markets[(pair, timeframe)] = market
            evaluated_by_market[market_key] = 0
            pip = PIP_SIZE.get(pair, 0.0001)
            spread_pips = float(quote.spread) / pip
            close_times = frame["candle_close_time"]
            trend = np.where(features.ema21 > features.ema50, "bullish",
                             np.where(features.ema21 < features.ema50, "bearish", "neutral"))
            feature_timelines[(pair, timeframe)] = pd.DataFrame(
                {"close_time": close_times, "trend_direction": trend})
            eligible = (features.groupby("segment", sort=False).cumcount().iloc[-1] >= STRATEGY_WARMUP
                        and _number(features.atr14.iloc[-1]) is not None)
            for strategy in strategies:
                evaluated += 1
                evaluated_by_market[market_key] += 1
                record: dict[str, Any] = {
                    "pair": pair, "timeframe": timeframe, "strategy_id": strategy.strategy_id,
                    "direction": "NO SIGNAL", "decision_timestamp": market["decision_timestamp"],
                    "evaluation_status": "evaluated", "error": None,
                }
                if not eligible:
                    record["evaluation_status"] = "insufficient_warmup"
                    evaluation_records.append(record)
                    continue
                try:
                    vector = strategy.signal_fn(features).fillna(0)
                    if len(vector) != len(features):
                        raise ValueError("strategy signal output length does not match candle history")
                    side = int(vector.iloc[-1])
                    if side not in (-1, 0, 1):
                        raise ValueError(f"unsupported strategy direction: {side}")
                except Exception as exc:
                    record["evaluation_status"] = "error"
                    record["error"] = str(exc)
                    evaluation_records.append(record)
                    errors.append({"pair": pair, "timeframe": timeframe,
                                   "strategy_id": strategy.strategy_id, "error": str(exc)})
                    continue
                if not side:
                    evaluation_records.append(record)
                    continue
                direction = "BUY" if side > 0 else "SELL"
                record["direction"] = direction
                evaluation_records.append(record)
                entry = float(quote.ask if side > 0 else quote.bid)
                atr = float(features.atr14.iloc[-1])
                distance = atr * 1.5
                stop = entry - side * distance
                target = entry + side * atr * 2.25
                selected_evidence, evidence_reasons, evidence_scope = _validation_evidence_for(
                    strategy.strategy_id, pair, timeframe, market["regime"], pair_evidence)
                expectancy = (_number(selected_evidence.get("expectancy_r",
                                                             selected_evidence.get("expectancy_net_r")))
                              if selected_evidence else None)
                trades = int(selected_evidence.get("trades", 0) or 0) if selected_evidence else 0
                cost_basis_pips = 3.5 if selected_evidence else None
                costs_complete = commission_pips_round_trip is not None and slippage_pips_round_trip is not None
                signal_decision = pd.Timestamp(market["decision_timestamp"])
                alignment = []
                missing_timeframes = []
                for candidate_tf in ("M15", "H1", "H4"):
                    timeline = feature_timelines.get((pair, candidate_tf))
                    if timeline is None:
                        missing_timeframes.append(candidate_tf)
                        continue
                    known = timeline.loc[timeline.close_time <= signal_decision]
                    if known.empty:
                        missing_timeframes.append(candidate_tf)
                        continue
                    latest = known.iloc[-1]
                    if signal_decision - latest.close_time > MAX_CANDLE_AGE[candidate_tf]:
                        missing_timeframes.append(candidate_tf)
                        continue
                    alignment.append({"timeframe": candidate_tf,
                                      "close_time": latest.close_time.isoformat(),
                                      "trend_direction": str(latest.trend_direction)})
                aligned = sum(item["trend_direction"] == direction.lower() for item in alignment)
                has_mtf = not missing_timeframes and len(alignment) == 3 and aligned >= 2
                signal = {"pair": pair, "timeframe": timeframe, "strategy_id": strategy.strategy_id,
                          "strategy_family": strategy.family, "direction": direction,
                          "entry_price": entry, "stop_loss": stop, "take_profit": target,
                          "risk_reward": 1.5, "atr14": atr, "regime": market["regime"],
                          "session_utc": market["session_utc"], "decision_timestamp": market["decision_timestamp"],
                          "spread_pips_observed": spread_pips,
                          "spread_observed_at": _utc_timestamp(
                              quote.time, "quote timestamp").isoformat(),
                          "commission_pips_round_trip": commission_pips_round_trip,
                          "slippage_pips_round_trip": slippage_pips_round_trip,
                          "expected_net_r": None, "validation_expectancy_net_r": expectancy,
                          "validation_evidence_scope": evidence_scope,
                          "validation_cost_assumption_pips_round_trip": cost_basis_pips,
                          "validation_trades": trades, "cost_inputs_complete": costs_complete,
                          "validation_rejection_reasons": evidence_reasons,
                          "evidence": strategy.explanation, "coordination_score": 0.0,
                          "paper_review_eligible": False, "rejection_reasons": [],
                          "multi_timeframe_trend_alignment_count": aligned,
                          "multi_timeframe_coverage": [item["timeframe"] for item in alignment],
                          "multi_timeframe_close_times": {item["timeframe"]: item["close_time"]
                                                          for item in alignment},
                          "missing_timeframes": missing_timeframes,
                          "multi_timeframe_as_of": signal_decision.isoformat()}
                signals.append(signal)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            market_rejections.append({"pair": pair, "timeframe": timeframe, "reason": str(exc)})

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for signal in signals:
        groups.setdefault((signal["pair"], signal["timeframe"], signal["direction"]), []).append(signal)
    conflicts = {(pair, tf) for pair, tf, _ in groups
                 if (pair, tf, "BUY") in groups and (pair, tf, "SELL") in groups}
    ranks: list[dict[str, Any]] = []
    for key, members in groups.items():
        pair, timeframe, direction = key
        members.sort(key=lambda row: row["strategy_id"])
        lead_member = max(members, key=lambda row: (
            row["validation_expectancy_net_r"] if row["validation_expectancy_net_r"] is not None else -np.inf,
            row["strategy_id"],
        ))
        lead = dict(lead_member)
        strategy_ids = [row["strategy_id"] for row in members]
        independent_bonus, independent, correlated, unknown = _independence_assessment(
            strategy_ids, correlations)
        mtf_count = lead.get("multi_timeframe_trend_alignment_count", 0)
        has_mtf = (not lead.get("missing_timeframes")
                   and len(lead.get("multi_timeframe_coverage", [])) == 3 and mtf_count >= 2)
        mtf_bonus = max(0, mtf_count - 1) * 0.5 if has_mtf else 0.0
        history_penalty = min(2.0, max(0.0, -(lead["validation_expectancy_net_r"] or 0.0) * 5.0))
        lead["supporting_strategies"] = [row["strategy_id"] for row in members
                                         if row["strategy_id"] != lead["strategy_id"]]
        lead["independent_confirmations"] = independent
        lead["correlated_confirmations"] = correlated
        lead["unknown_independence_confirmations"] = unknown
        lead["independence_evidence_scope"] = "Mission 11 validation-only daily exit-return correlations"
        lead["independence_uncertainty"] = (
            "Correlation is unknown for one or more supporting strategies; no independence credit was awarded."
            if unknown else None)
        lead["opposing_signals"] = ["BUY" if direction == "SELL" else "SELL"] if (pair, timeframe) in conflicts else []
        lead["coordination_score"] = round(1.0 + independent_bonus + mtf_bonus - history_penalty, 4)
        reasons = []
        if (pair, timeframe) in conflicts:
            reasons.append("conflicting BUY and SELL strategy signals on this pair/timeframe")
        reasons.extend(lead.get("validation_rejection_reasons", []))
        if not lead["validation_rejection_reasons"] and lead["validation_expectancy_net_r"] is None:
            reasons.append("compatible validation expectancy is unavailable")
        if not lead["cost_inputs_complete"]:
            reasons.append("commission and slippage cost inputs are not calibrated")
        if portfolio_exposure_clear is not True:
            reasons.append("portfolio exposure is open or could not be verified")
        if not has_mtf:
            if lead.get("missing_timeframes"):
                reasons.append("missing or not-yet-closed timeframe data: " +
                               ", ".join(lead["missing_timeframes"]))
            else:
                reasons.append("no timeframe trend alignment at the signal decision timestamp")
        if lead["cost_inputs_complete"] and lead["validation_expectancy_net_r"] is not None:
            risk_pips = lead["atr14"] * 1.5 / PIP_SIZE.get(pair, 0.0001)
            if risk_pips > 0 and lead["validation_cost_assumption_pips_round_trip"] is not None:
                live_costs = (lead["spread_pips_observed"] + commission_pips_round_trip + slippage_pips_round_trip)
                lead["expected_net_r"] = lead["validation_expectancy_net_r"] + (
                    lead["validation_cost_assumption_pips_round_trip"] - live_costs) / risk_pips
                if lead["expected_net_r"] <= 0:
                    reasons.append("cost-adjusted validation expectancy is nonpositive")
        else:
            reasons.append("expected net R unavailable until pair/timeframe validation and complete cost inputs exist")
        lead["rejection_reasons"] = reasons
        lead["paper_review_eligible"] = not reasons
        ranks.append(lead)
    ranks.sort(key=lambda row: (row["paper_review_eligible"], row["coordination_score"],
                                row["validation_expectancy_net_r"] or -999.0), reverse=True)
    selected = next((row for row in ranks if row["paper_review_eligible"]), None)
    market_payload = [{"pair": pair, **value} for (pair, _), value in sorted(markets.items())]
    return {"mode": "research_paper_only", "read_only": True, "live_trading": False,
            "model_probabilities": None, "strategies_available": len(strategies),
            "portfolio_exposure_clear": portfolio_exposure_clear,
            "portfolio_risk_note": "Read-only open-position check only; this research coordinator does not size or approve account risk.",
            "as_of_utc": as_of.isoformat(),
            "strategy_ids": [strategy.strategy_id for strategy in strategies],
            "markets_evaluated": len(markets),
            "strategy_evaluations": evaluated,
            "evaluations_per_market": (len(strategies) if evaluated_by_market and
                                       all(count == len(strategies) for count in evaluated_by_market.values()) else 0),
            "market_evaluation_counts": evaluated_by_market,
            "strategy_evaluation_records": evaluation_records,
            "market_intelligence": market_payload, "signals": signals,
            "ranked_opportunities": ranks, "decision": selected,
            "status": "opportunity" if selected else "no_trade",
            "reason": "Paper candidate passed all evidence and cost gates." if selected else
                      "NO TRADE: no candidate passed historical validation, conflict, multi-timeframe, and cost gates.",
            "evidence_source": "Mission 9 pair/timeframe/regime validation rows only; no pooled or test-split evidence is used.",
            "legacy_pooled_evidence_supplied_but_ignored": validation_evidence is not None,
            "evidence_cost_assumption": "Mission 9 comparable metrics use the declared 3.5 pip round-trip cost scenario; broker costs remain uncalibrated.",
            "independence_source": "Mission 11 validation-only daily exit-return correlations; unknown correlations receive no independence bonus.",
            "errors": errors, "market_rejections": market_rejections}
