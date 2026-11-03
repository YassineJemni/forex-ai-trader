"""Reproducible, event-driven multi-strategy research on stored MT5 candles.

This module reads original MT5 data only. The only dataset writes are new,
idempotent derived EURUSD/USDJPY H4 research files and a quarantined USDJPY
pre-1999 segment. It never connects to MT5 or submits orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data_integrity import OHLC, TIMEFRAME_MINUTES


ROOT = Path(__file__).resolve().parents[2]
PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")
TIMEFRAMES = ("M15", "H1", "H4")
PIP_SIZE = {pair: 0.01 if pair.endswith("JPY") else 0.0001 for pair in PAIRS}
STRATEGY_WARMUP = 100
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)


@dataclass(frozen=True)
class ResearchConfig:
    account_balance: float = 10_000.0  # transparent research notional, not broker account state
    risk_fraction: float = 0.005
    max_open_positions: int = 1
    allow_overlapping: bool = False
    stop_atr_multiple: float = 1.5
    target_atr_multiple: float = 2.25
    spread_pips: float | None = None
    commission_pips_round_trip: float | None = None
    slippage_pips_round_trip: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.account_balance) or self.account_balance <= 0:
            raise ValueError("account_balance must be finite and positive")
        if not 0 < self.risk_fraction <= 0.02:
            raise ValueError("risk_fraction must be in (0, 0.02]")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1")
        if self.stop_atr_multiple <= 0 or self.target_atr_multiple <= 0:
            raise ValueError("stop and target ATR multiples must be positive")
        for name in ("spread_pips", "commission_pips_round_trip", "slippage_pips_round_trip"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative when specified")

    @property
    def costs_complete(self) -> bool:
        return all(getattr(self, name) is not None for name in (
            "spread_pips", "commission_pips_round_trip", "slippage_pips_round_trip"))

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["net_results_status"] = "assumption-based" if self.costs_complete else "unverified: one or more cost inputs absent"
        value["account_balance_is_research_notional"] = True
        return value


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    family: str
    explanation: str
    signal_fn: Callable[[pd.DataFrame], pd.Series]


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _signal(mask_long: pd.Series, mask_short: pd.Series) -> pd.Series:
    result = pd.Series(0, index=mask_long.index, dtype="int8")
    result.loc[mask_long.fillna(False) & ~mask_short.fillna(False)] = 1
    result.loc[mask_short.fillna(False) & ~mask_long.fillna(False)] = -1
    return result


def strategy_library() -> tuple[Strategy, ...]:
    """Independent, deterministic hypotheses. No indicator uses future rows."""
    def ema_cross(f): return _signal(_cross_up(f.ema9, f.ema21), _cross_down(f.ema9, f.ema21))
    def trend_pullback(f):
        return _signal((f.ema21 > f.ema50) & (f.low <= f.ema21) & (f.close > f.ema9) & (f.momentum5 > 0),
                       (f.ema21 < f.ema50) & (f.high >= f.ema21) & (f.close < f.ema9) & (f.momentum5 < 0))
    def donchian(f): return _signal(f.close > f.prior_high20, f.close < f.prior_low20)
    def breakout_retest(f):
        up = f.close.gt(f.prior_high20).shift(1).rolling(3, min_periods=1).max().fillna(0).gt(0) & (f.low <= f.prior_high20) & (f.close > f.prior_high20)
        down = f.close.lt(f.prior_low20).shift(1).rolling(3, min_periods=1).max().fillna(0).gt(0) & (f.high >= f.prior_low20) & (f.close < f.prior_low20)
        return _signal(up, down)
    def rsi_reversal(f): return _signal(_cross_up(f.rsi14, pd.Series(30.0, index=f.index)), _cross_down(f.rsi14, pd.Series(70.0, index=f.index)))
    def bollinger_reversal(f): return _signal((f.close.shift(1) < f.bb_lower.shift(1)) & (f.close >= f.bb_lower), (f.close.shift(1) > f.bb_upper.shift(1)) & (f.close <= f.bb_upper))
    def market_structure(f): return _signal((f.close > f.prior_swing_high) & (f.ema21 > f.ema50), (f.close < f.prior_swing_low) & (f.ema21 < f.ema50))
    def structure_choch(f): return _signal((f.close > f.prior_swing_high) & (f.structure_trend < 0), (f.close < f.prior_swing_low) & (f.structure_trend > 0))
    def structure_hh_hl(f): return _signal(f.higher_high & f.higher_low, f.lower_high & f.lower_low)
    def liquidity_sweep(f): return _signal((f.low < f.prior_low10) & (f.close > f.prior_low10), (f.high > f.prior_high10) & (f.close < f.prior_high10))
    def fvg_hypothesis(f): return _signal((f.low > f.high.shift(2)) & (f.close > f.open) & (f.close > f.ema21), (f.high < f.low.shift(2)) & (f.close < f.open) & (f.close < f.ema21))
    def order_block_hypothesis(f): return _signal((f.close > f.prior_high10) & (f.prior_bearish & (f.low <= f.prior_bearish_high)),
                                                   (f.close < f.prior_low10) & (f.prior_bullish & (f.high >= f.prior_bullish_low)) )
    def displacement_rejection(f): return _signal((f.lower_wick > 2 * f.body.abs()) & (f.close > f.open) & (f.low <= f.prior_low20),
                                                   (f.upper_wick > 2 * f.body.abs()) & (f.close < f.open) & (f.high >= f.prior_high20))
    def engulfing(f): return _signal((f.close > f.open) & (f.close.shift(1) < f.open.shift(1)) & (f.close >= f.open.shift(1)) & (f.open <= f.close.shift(1)) & (f.low <= f.prior_low20),
                                     (f.close < f.open) & (f.close.shift(1) > f.open.shift(1)) & (f.close <= f.open.shift(1)) & (f.open >= f.close.shift(1)) & (f.high >= f.prior_high20))
    return (
        Strategy("ema_crossover", "trend_following", "EMA 9/21 crossover after candle close", ema_cross),
        Strategy("trend_pullback_momentum", "trend_following", "EMA 21/50 alignment, EMA 21 pullback, and 5-bar momentum", trend_pullback),
        Strategy("donchian_breakout", "breakout", "Close beyond prior 20-bar high/low with ATR volatility context", donchian),
        Strategy("breakout_retest", "breakout", "Prior 3-bar breakout followed by close-confirmed level retest", breakout_retest),
        Strategy("rsi14_recross", "mean_reversion", "RSI 14 recross out of 30/70 extremes", rsi_reversal),
        Strategy("bollinger_reentry", "mean_reversion", "Close re-enters 20-bar, 2-sigma Bollinger envelope", bollinger_reversal),
        Strategy("structure_bos", "market_structure", "Close breaks prior 20-bar swing boundary with EMA trend alignment", market_structure),
        Strategy("structure_choch", "market_structure", "BOS counter to the previously confirmed swing trend (CHoCH hypothesis)", structure_choch),
        Strategy("structure_hh_hl", "market_structure", "Sequence of confirmed higher highs/higher lows or lower highs/lower lows", structure_hh_hl),
        Strategy("liquidity_sweep", "liquidity_smc_hypothesis", "Prior 10-bar high/low sweep with close back inside", liquidity_sweep),
        Strategy("fair_value_gap", "liquidity_smc_hypothesis", "Three-candle imbalance with directional close confirmation", fvg_hypothesis),
        Strategy("order_block_hypothesis", "liquidity_smc_hypothesis", "Prior opposing candle retest on 10-bar structure break", order_block_hypothesis),
        Strategy("displacement_rejection", "price_action", "Long wick rejection at 20-bar boundary", displacement_rejection),
        Strategy("engulfing_at_level", "price_action", "Engulfing candle at prior 20-bar boundary", engulfing),
    )


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["time", *OHLC]
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if _frame_fingerprint(existing) != _frame_fingerprint(frame):
            raise FileExistsError(f"refusing to overwrite nonmatching derived data: {path}")
        return
    frame.to_parquet(path, index=False)


def _write_manifest_once(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("source_file", "source_fingerprint_sha256", "derived_fingerprint_sha256", "filter", "rows"):
            if existing.get(key) != manifest.get(key):
                raise FileExistsError(f"refusing to replace nonmatching derived provenance: {path}")
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_research_frame(pair: str, timeframe: str, data_dir: Path | None = None, *,
                        derived_dir: Path | None = None, quarantine_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if pair not in PAIRS or timeframe not in TIMEFRAMES:
        raise ValueError("unsupported pair/timeframe")
    root = data_dir or ROOT / "market-data" / "mt5-history"
    source_path = root / f"{pair}_{timeframe}.parquet"
    frame = pd.read_parquet(source_path)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise ValueError(f"duplicate or unordered timestamps in {source_path}")
    px = frame.loc[:, OHLC].to_numpy(dtype=float)
    if not np.isfinite(px).all() or (px <= 0).any() or (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"invalid OHLC values in {source_path}")
    original_rows = len(frame)
    exclusions: dict[str, Any] = {"source_file": str(source_path), "excluded_rows": 0, "exclusion_reason": None}
    if timeframe == "H4" and pair == "EURUSD":
        mask = frame.time < pd.Timestamp("1999-01-01", tz="UTC")
        exclusions.update({"excluded_rows": int(mask.sum()), "excluded_start_utc": frame.loc[mask, "time"].min().isoformat() if mask.any() else None,
                           "excluded_end_utc": frame.loc[mask, "time"].max().isoformat() if mask.any() else None,
                           "exclusion_reason": "pre-euro observations cannot be EURUSD"})
        frame = frame.loc[~mask].reset_index(drop=True)
        derived = (derived_dir or ROOT / "market-data" / "derived-research") / f"{pair}_{timeframe}.parquet"
        _atomic_parquet(frame, derived)
        exclusions["derived_file"] = str(derived)
        _write_manifest_once(derived.with_suffix(".provenance.json"), {
            "source_file": str(source_path), "source_fingerprint_sha256": _frame_fingerprint(pd.read_parquet(source_path)),
            "derived_fingerprint_sha256": _frame_fingerprint(frame), "filter": "time >= 1999-01-01T00:00:00Z",
            "exclusion_reason": exclusions["exclusion_reason"], "rows": len(frame), "timezone": "UTC",
        })
    elif timeframe == "H4" and pair == "USDJPY":
        mask = frame.time < pd.Timestamp("1999-01-01", tz="UTC")
        suspect = frame.loc[mask].copy()
        exclusions.update({"excluded_rows": int(mask.sum()), "excluded_start_utc": suspect.time.min().isoformat() if len(suspect) else None,
                           "excluded_end_utc": suspect.time.max().isoformat() if len(suspect) else None,
                           "exclusion_reason": "pre-1999 daily-sampled rows quarantined pending broker provenance"})
        quarantine = (quarantine_dir or ROOT / "market-data" / "quarantine") / "USDJPY_H4_pre1999.parquet"
        if len(suspect):
            _atomic_parquet(suspect, quarantine)
            _write_manifest_once(quarantine.with_suffix(".provenance.json"), {
                "source_file": str(source_path), "source_fingerprint_sha256": _frame_fingerprint(pd.read_parquet(source_path)),
                "derived_fingerprint_sha256": _frame_fingerprint(suspect), "filter": "time < 1999-01-01T00:00:00Z",
                "exclusion_reason": exclusions["exclusion_reason"], "rows": len(suspect), "timezone": "UTC",
            })
        exclusions["quarantine_file"] = str(quarantine)
        frame = frame.loc[~mask].reset_index(drop=True)
        derived = (derived_dir or ROOT / "market-data" / "derived-research") / f"{pair}_{timeframe}.parquet"
        _atomic_parquet(frame, derived)
        exclusions["derived_file"] = str(derived)
        _write_manifest_once(derived.with_suffix(".provenance.json"), {
            "source_file": str(source_path), "source_fingerprint_sha256": _frame_fingerprint(pd.read_parquet(source_path)),
            "derived_fingerprint_sha256": _frame_fingerprint(frame), "filter": "time >= 1999-01-01T00:00:00Z",
            "exclusion_reason": exclusions["exclusion_reason"], "rows": len(frame), "timezone": "UTC",
        })
    exclusions["original_rows"] = original_rows
    exclusions["research_rows"] = len(frame)
    exclusions["research_fingerprint_sha256"] = _frame_fingerprint(frame)
    exclusions["timezone"] = str(frame.time.dtype.tz)
    return frame.reset_index(drop=True), exclusions


def build_features(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Causal features that retain state through weekends, reset after >3 days."""
    result = frame.copy().reset_index(drop=True)
    gaps = result.time.diff()
    # Carry indicator state through the normal weekly FX closure. Only reset
    # after a multi-day outage, independent of timeframe.
    segment = gaps.gt(pd.Timedelta(days=3)).cumsum()
    result["segment"] = segment.astype("int32")
    groups = result.groupby("segment", sort=False)
    close, high, low = result.close, result.high, result.low
    for span, name in ((9, "ema9"), (21, "ema21"), (50, "ema50")):
        result[name] = groups.close.transform(lambda s: s.ewm(span=span, adjust=False).mean())
    previous_close = groups.close.shift(1)
    tr = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    result["atr14"] = tr.groupby(segment, sort=False).transform(lambda s: s.rolling(14, min_periods=14).mean())
    change = groups.close.diff()
    gain = change.clip(lower=0).groupby(segment, sort=False).transform(lambda s: s.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    loss = (-change.clip(upper=0)).groupby(segment, sort=False).transform(lambda s: s.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    result["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    for window in (10, 20):
        result[f"prior_high{window}"] = groups.high.transform(lambda s: s.shift(1).rolling(window, min_periods=window).max())
        result[f"prior_low{window}"] = groups.low.transform(lambda s: s.shift(1).rolling(window, min_periods=window).min())
    result["bb_mid"] = groups.close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    std = groups.close.transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    result["bb_upper"], result["bb_lower"] = result.bb_mid + 2*std, result.bb_mid - 2*std
    result["momentum5"] = groups.close.pct_change(5)
    result["prior_swing_high"], result["prior_swing_low"] = result.prior_high20, result.prior_low20
    # A five-bar pivot is only recognized two bars after its center. The
    # confirmation row and all subsequent features therefore use no future bar.
    confirmed_high = groups.high.transform(lambda s: s.shift(2).eq(s.rolling(5, min_periods=5).max()))
    confirmed_low = groups.low.transform(lambda s: s.shift(2).eq(s.rolling(5, min_periods=5).min()))
    high_values = groups.high.shift(2).where(confirmed_high)
    low_values = groups.low.shift(2).where(confirmed_low)
    result["confirmed_swing_high"] = high_values
    result["confirmed_swing_low"] = low_values
    result["prior_confirmed_swing_high"] = high_values.groupby(segment, sort=False).transform(lambda s: s.ffill().shift(1))
    result["prior_confirmed_swing_low"] = low_values.groupby(segment, sort=False).transform(lambda s: s.ffill().shift(1))
    result["higher_high"] = confirmed_high & (high_values > high_values.groupby(segment, sort=False).transform(lambda s: s.ffill().shift(1)))
    result["lower_high"] = confirmed_high & (high_values < high_values.groupby(segment, sort=False).transform(lambda s: s.ffill().shift(1)))
    result["higher_low"] = confirmed_low & (low_values > low_values.groupby(segment, sort=False).transform(lambda s: s.ffill().shift(1)))
    result["lower_low"] = confirmed_low & (low_values < low_values.groupby(segment, sort=False).transform(lambda s: s.ffill().shift(1)))
    result["structure_trend"] = np.sign((result.ema21 - result.ema50).groupby(segment, sort=False).shift(1))
    result["prior_bearish"] = groups.open.shift(1) > groups.close.shift(1)
    result["prior_bullish"] = groups.close.shift(1) > groups.open.shift(1)
    result["prior_bearish_high"] = groups.high.shift(1).where(result.prior_bearish)
    result["prior_bullish_low"] = groups.low.shift(1).where(result.prior_bullish)
    result["body"] = result.close - result.open
    result["upper_wick"] = result.high - result[["open", "close"]].max(axis=1)
    result["lower_wick"] = result[["open", "close"]].min(axis=1) - result.low
    atr_pct = result.atr14 / result.close
    vol_median = groups["atr14"].transform(lambda s: (s / result.loc[s.index, "close"]).rolling(100, min_periods=30).median())
    trend_strength = (result.ema21 - result.ema50).abs() / result.atr14.replace(0, np.nan)
    result["regime"] = "transitional_uncertain"
    result.loc[atr_pct < vol_median * 0.7, "regime"] = "low_volatility"
    result.loc[atr_pct > vol_median * 1.5, "regime"] = "high_volatility"
    result.loc[(trend_strength > 1.0) & (atr_pct.between(vol_median * 0.7, vol_median * 1.5)), "regime"] = "trending"
    result.loc[(trend_strength <= 1.0) & (atr_pct.between(vol_median * 0.7, vol_median * 1.5)), "regime"] = "ranging"
    return result.replace([np.inf, -np.inf], np.nan)


def generate_signals(features: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    direction = strategy.signal_fn(features).fillna(0).astype("int8")
    eligible = features.groupby("segment", sort=False).cumcount().ge(STRATEGY_WARMUP)
    eligible &= features.atr14.notna()
    indices = np.flatnonzero((direction.ne(0) & eligible).to_numpy())
    return pd.DataFrame({
        "source_index": indices,
        "decision_timestamp": features.time.iloc[indices].to_numpy(),
        "direction": direction.iloc[indices].to_numpy(),
        "entry_logic": "next bar open after decision candle close",
        "invalidation_logic": "ATR(14) stop with configured multiple; stop has priority if SL and TP both touch",
        "target_logic": "ATR(14) target with configured multiple",
        "regime": features.regime.iloc[indices].to_numpy(),
        "evidence": strategy.explanation,
    })


def _session(timestamp: pd.Timestamp) -> str:
    hour = timestamp.hour
    return "Asia_UTC_00_06" if hour < 7 else "London_UTC_07_12" if hour < 13 else "NewYork_UTC_13_20" if hour < 21 else "Rollover_UTC_21_23"


def simulate_trades(frame: pd.DataFrame, features: pd.DataFrame, signals: pd.DataFrame,
                    pair: str, timeframe: str, config: ResearchConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sequential next-open entry; conservative stop-first intrabar collision rule."""
    rows: list[dict[str, Any]] = []
    rejects = {"no_next_bar": 0, "invalid_atr": 0, "overlap_skipped": 0, "boundary_open": 0, "boundary_close": 0}
    times = pd.to_datetime(frame.time, utc=True)
    pip = PIP_SIZE[pair]
    active_until: list[int] = []
    for signal in signals.itertuples(index=False):
        decision_idx = int(signal.source_index)
        entry_idx = decision_idx + 1
        if entry_idx >= len(frame):
            rejects["no_next_bar"] += 1
            continue
        active_until = [exit_idx for exit_idx in active_until if exit_idx >= entry_idx]
        if (not config.allow_overlapping and active_until) or len(active_until) >= config.max_open_positions:
            rejects["overlap_skipped"] += 1
            continue
        atr = float(features.atr14.iloc[decision_idx])
        if not np.isfinite(atr) or atr <= 0:
            rejects["invalid_atr"] += 1
            continue
        side = int(signal.direction)
        entry = float(frame.open.iloc[entry_idx])
        distance = atr * config.stop_atr_multiple
        stop = entry - side * distance
        target = entry + side * atr * config.target_atr_multiple
        quote_factor = 1.0 if pair.endswith("USD") else 1.0 / entry  # quote-to-USD conversion for USD/JPY
        stop_pips = distance / pip
        units = config.account_balance * config.risk_fraction / (distance * quote_factor)
        exit_idx, exit_price, exit_reason = len(frame) - 1, float(frame.close.iloc[-1]), "data_end"
        for j in range(entry_idx, len(frame)):
            bar_high, bar_low, bar_open = float(frame.high.iloc[j]), float(frame.low.iloc[j]), float(frame.open.iloc[j])
            stop_hit = bar_low <= stop if side > 0 else bar_high >= stop
            target_hit = bar_high >= target if side > 0 else bar_low <= target
            if stop_hit and target_hit:
                exit_idx, exit_price, exit_reason = j, (min(bar_open, stop) if side > 0 else max(bar_open, stop)), "stop_and_target_same_bar_stop_first"
                break
            if stop_hit:
                exit_idx, exit_price = j, (min(bar_open, stop) if side > 0 else max(bar_open, stop))
                exit_reason = "stop_loss"
                break
            if target_hit:
                exit_idx, exit_price, exit_reason = j, target, "take_profit"
                break
        gross_pips = side * (exit_price - entry) / pip
        costs = None
        net = None
        if config.costs_complete:
            costs = float(config.spread_pips + config.commission_pips_round_trip + config.slippage_pips_round_trip)
            net = gross_pips - costs
        entry_time = times.iloc[entry_idx]
        exit_time = times.iloc[exit_idx] + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        active_until.append(exit_idx)
        rows.append({"strategy_id": None, "pair": pair, "timeframe": timeframe,
                     "decision_timestamp": times.iloc[decision_idx].isoformat(),
                     "entry_timestamp": entry_time.isoformat(), "exit_timestamp": exit_time.isoformat(),
                     "direction": "BUY" if side > 0 else "SELL", "entry_price": entry,
                     "stop_loss": stop, "take_profit": target, "exit_price": exit_price,
                     "exit_reason": exit_reason, "gross_pnl_pips": gross_pips,
                     "costs_pips": costs, "net_pnl_pips": net, "stop_distance_pips": stop_pips,
                     "risk_reward": config.target_atr_multiple / config.stop_atr_multiple,
                     "position_units": units, "gross_pnl_account_currency": gross_pips * pip * units * quote_factor,
                     "risk_amount": config.account_balance * config.risk_fraction,
                     "regime": signal.regime, "session_utc": _session(entry_time), "signal_evidence": signal.evidence})
    return pd.DataFrame(rows), rejects


def chronological_splits(frame: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    times = pd.to_datetime(frame.time, utc=True)
    n = len(times)
    i1, i2 = int(n * SPLIT_FRACTIONS[0]), int(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    return {"train": (times.iloc[0], times.iloc[i1]), "validation": (times.iloc[i1], times.iloc[i2]),
            "test": (times.iloc[i2], times.iloc[-1] + pd.Timedelta(microseconds=1))}


def summarize_trades(trades: pd.DataFrame, *, costs_verified: bool) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0, "win_rate": None, "gross_profit_pips": 0.0, "net_profit_pips": None,
                "profit_factor_gross": None, "expectancy_gross_pips": None, "max_drawdown_gross_pips": 0.0,
                "average_win_pips": None, "average_loss_pips": None, "max_consecutive_losses": 0, "sharpe_gross": None}
    pnl = trades.gross_pnl_pips.astype(float)
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    pf = float(winners.sum() / abs(losers.sum())) if len(losers) and losers.sum() else None
    equity = pnl.cumsum()
    drawdown = equity.cummax().clip(lower=0) - equity
    consec = best = 0
    for loss in pnl.lt(0):
        consec = consec + 1 if loss else 0
        best = max(best, consec)
    net = float(trades.net_pnl_pips.sum()) if costs_verified else None
    returns_std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    return {"trades": len(trades), "win_rate": float(pnl.gt(0).mean()), "gross_profit_pips": float(pnl.sum()),
            "net_profit_pips": net, "profit_factor_gross": pf, "expectancy_gross_pips": float(pnl.mean()),
            "max_drawdown_gross_pips": float(drawdown.max()), "average_win_pips": float(winners.mean()) if len(winners) else None,
            "average_loss_pips": float(losers.mean()) if len(losers) else None, "max_consecutive_losses": best,
            "sharpe_gross": float(pnl.mean() / returns_std * np.sqrt(len(pnl))) if returns_std else None,
            "average_realized_risk_reward": float(winners.mean() / abs(losers.mean())) if len(winners) and len(losers) and losers.mean() else None}


def evaluate_combo(pair: str, timeframe: str, frame: pd.DataFrame, config: ResearchConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    features = build_features(frame, timeframe)
    splits = chronological_splits(frame)
    results: list[dict[str, Any]] = []
    trade_logs: list[dict[str, Any]] = []
    signal_vectors: dict[str, pd.Series] = {}
    for strategy in strategy_library():
        signals = generate_signals(features, strategy)
        vector = pd.Series(0, index=np.arange(len(frame)), dtype="int8")
        if not signals.empty:
            vector.iloc[signals.source_index.to_numpy(dtype=int)] = signals.direction.to_numpy(dtype="int8")
        signal_vectors[strategy.strategy_id] = vector
        trades, rejected = simulate_trades(frame, features, signals, pair, timeframe, config)
        if not trades.empty:
            trades["strategy_id"] = strategy.strategy_id
            trades["strategy_family"] = strategy.family
            for row in trades.to_dict("records"):
                trade_logs.append(row)
        split_metrics = {}
        for name, (start, end) in splits.items():
            in_period = (pd.to_datetime(trades.entry_timestamp, utc=True) >= start) & (pd.to_datetime(trades.exit_timestamp, utc=True) < end) if len(trades) else pd.Series([], dtype=bool)
            subset = trades.loc[in_period] if len(trades) else trades
            split_metrics[name] = summarize_trades(subset, costs_verified=config.costs_complete)
            split_metrics[name]["purged_boundary_crossing_or_outside_trades"] = int(len(trades) - in_period.sum()) if len(trades) else 0
        result = {"strategy_id": strategy.strategy_id, "family": strategy.family, "pair": pair, "timeframe": timeframe,
                  "bars": len(frame), "signals": len(signals), "rejections": rejected,
                  "regime_counts": trades.regime.value_counts().to_dict() if len(trades) else {},
                  "session_counts": trades.session_utc.value_counts().to_dict() if len(trades) else {},
                  "regime_performance": {str(k): summarize_trades(v, costs_verified=config.costs_complete) for k, v in trades.groupby("regime")} if len(trades) else {},
                  "session_performance_utc": {str(k): summarize_trades(v, costs_verified=config.costs_complete) for k, v in trades.groupby("session_utc")} if len(trades) else {},
                  "splits": split_metrics, "hypothesis": strategy.explanation}
        results.append(result)
    relationships = []
    ids = list(signal_vectors)
    for i, left in enumerate(ids):
        a = signal_vectors[left].to_numpy()
        for right in ids[i + 1:]:
            b = signal_vectors[right].to_numpy()
            active = (a != 0) | (b != 0)
            both = (a != 0) & (b != 0)
            agreement = int(((a == b) & both).sum())
            relationships.append({"pair": pair, "timeframe": timeframe, "strategy_a": left, "strategy_b": right,
                                  "signals_a": int((a != 0).sum()), "signals_b": int((b != 0).sum()),
                                  "coincident_timestamps": int(both.sum()),
                                  "directional_agreement_when_coincident": agreement / int(both.sum()) if both.any() else None,
                                  "signal_overlap_share_of_smaller": int(both.sum()) / max(1, min(int((a != 0).sum()), int((b != 0).sum()))),
                                  "interpretation": "potentially redundant" if both.any() and agreement / int(both.sum()) >= 0.8 else "overlap evidence only; complementarity not established"})
    return results, trade_logs, relationships


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if pd.isna(value): return None
    raise TypeError(f"not serializable: {type(value).__name__}")


def run_research(config: ResearchConfig | None = None, *, output_dir: Path | None = None,
                 data_dir: Path | None = None) -> dict[str, Any]:
    config = config or ResearchConfig()
    out = output_dir or ROOT / "reports" / "strategy-research"
    out.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    data_exclusions: list[dict[str, Any]] = []
    signal_relationships: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for pair in PAIRS:
        for timeframe in TIMEFRAMES:
            frame, provenance = load_research_frame(pair, timeframe, data_dir)
            data_exclusions.append(provenance)
            fingerprints[f"{pair}_{timeframe}"] = provenance["research_fingerprint_sha256"]
            print(f"START {pair} {timeframe} bars={len(frame)}", flush=True)
            combo_results, trades, relationships = evaluate_combo(pair, timeframe, frame, config)
            all_results.extend(combo_results)
            all_trades.extend(trades)
            signal_relationships.extend(relationships)
            print(f"DONE  {pair} {timeframe} strategies={len(combo_results)} trades={len(trades)}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trades_frame = pd.DataFrame(all_trades)
    trade_path = out / f"trade_log_{stamp}.csv"
    trades_frame.to_csv(trade_path, index=False)
    curve_path = out / f"equity_drawdown_{stamp}.csv"
    if not trades_frame.empty:
        curves = []
        for key, part in trades_frame.groupby(["strategy_id", "pair", "timeframe"], sort=True):
            part = part.sort_values("exit_timestamp").copy()
            part["equity_gross_pips"] = part.gross_pnl_pips.cumsum()
            part["drawdown_gross_pips"] = part.equity_gross_pips.cummax() - part.equity_gross_pips
            curves.append(part[["strategy_id", "pair", "timeframe", "exit_timestamp", "equity_gross_pips", "drawdown_gross_pips"]])
        pd.concat(curves, ignore_index=True).to_csv(curve_path, index=False)
    else:
        pd.DataFrame(columns=["strategy_id", "pair", "timeframe", "exit_timestamp", "equity_gross_pips", "drawdown_gross_pips"]).to_csv(curve_path, index=False)
    # Strategy-level aggregate is descriptive only. No winner is promoted and
    # unverified gross results are not presented as a profitability ranking.
    leaderboard = []
    for strategy in strategy_library():
        items = [r for r in all_results if r["strategy_id"] == strategy.strategy_id]
        leaderboard.append({"strategy_id": strategy.strategy_id, "family": strategy.family,
                            "pair_timeframe_combinations": len(items),
                            "validation_trades": sum(x["splits"]["validation"]["trades"] for x in items),
                            "validation_gross_pips_sum": sum(x["splits"]["validation"]["gross_profit_pips"] for x in items),
                            "cost_status": "unverified" if not config.costs_complete else "assumption-based"})
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(),
              "pairs": list(PAIRS), "timeframes": list(TIMEFRAMES), "pair_timeframe_combinations": len(fingerprints),
              "strategies": len(strategy_library()), "strategy_runs": len(all_results),
              "data_exclusions_and_provenance": data_exclusions, "data_fingerprints": fingerprints,
              "results": all_results, "leaderboard_descriptive_only": leaderboard,
              "signal_relationships": signal_relationships,
              "gross_only_when_costs_missing": not config.costs_complete,
              "trade_log": str(trade_path), "equity_drawdown_curves": str(curve_path),
              "training_or_model_promotion": False, "broker_or_order_activity": False}
    json_path = out / f"strategy_research_{stamp}.json"
    md_path = out / f"strategy_research_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"REPORT {md_path}\nJSON {json_path}\nTRADE_LOG {trade_path}\nCURVES {curve_path}", flush=True)
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Multi-Strategy Forex Research", "", f"Generated: {report['generated_at_utc']}", "",
             f"Combos: {report['pair_timeframe_combinations']} | Strategies: {report['strategies']} | Runs: {report['strategy_runs']}", "",
             "Net performance is unverified because historical spread, commission, and slippage inputs were not supplied." if report["gross_only_when_costs_missing"] else "Net results use declared assumptions; these are not broker-calibrated.",
             "The account balance is an explicit research notional. No model was trained or promoted. No strategy was selected for live use.", "",
             "## Descriptive validation leaderboard", "", "| Strategy | Family | Validation trades | Gross validation pips (sum across combos) | Cost status |", "|---|---|---:|---:|---|"]
    for row in report["leaderboard_descriptive_only"]:
        lines.append(f"| {row['strategy_id']} | {row['family']} | {row['validation_trades']} | {row['validation_gross_pips_sum']:.2f} | {row['cost_status']} |")
    lines += ["", "These aggregate gross figures mix instruments and timeframes and are descriptive only; they are not a ranking for selection.",
              "", "## Per-combination / split metrics", "", "| Strategy | Pair | TF | Split | Trades | Win % | Gross pips | PF gross | Expectancy | Max DD | Max losses |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in report["results"]:
        for split, m in result["splits"].items():
            win = "n/a" if m["win_rate"] is None else f"{m['win_rate']:.1%}"
            pf = "n/a" if m["profit_factor_gross"] is None else f"{m['profit_factor_gross']:.3f}"
            ex = "n/a" if m["expectancy_gross_pips"] is None else f"{m['expectancy_gross_pips']:.3f}"
            lines.append(f"| {result['strategy_id']} | {result['pair']} | {result['timeframe']} | {split} | {m['trades']} | {win} | {m['gross_profit_pips']:.2f} | {pf} | {ex} | {m['max_drawdown_gross_pips']:.2f} | {m['max_consecutive_losses']} |")
    lines += ["", "## Dataset exclusions", ""]
    for item in report["data_exclusions_and_provenance"]:
        if item["excluded_rows"]:
            lines.append(f"- {item['source_file']}: excluded {item['excluded_rows']} rows ({item['excluded_start_utc']} through {item['excluded_end_utc']}); {item['exclusion_reason']}. Derived/quarantine: {item.get('derived_file', item.get('quarantine_file'))}.")
    lines += ["", "## Assumptions and limits", "",
              "- Signals are formed only from the current and past closed candles; entries occur at the next candle open.",
              "- Stop/target checks use bar OHLC. If both are touched in one candle, stop loss is applied first. Intrabar ordering is unknowable from OHLC.",
              "- Trades crossing chronological split boundaries are purged from split metrics. Indicators may use prior-period candles, as expected in forward evaluation.",
              "- Costs are disabled until spread, commission, and slippage are explicitly supplied. Gross uses the stored broker chart series; net profitability is unverified.",
              "- Currency session labels are UTC hour buckets, not broker-local session calendars. Regime labels use causal ATR and EMA heuristics.",
              "- SMC, order-block, FVG, BOS, and candle-pattern strategies are research hypotheses only.",
              "- Strategy signals are evaluated independently; there is no portfolio combination, AI selection, or live recommendation.",
              "- Full trades, curves, exact data fingerprints, row exclusions, configs, and per-combination results are in the companion JSON/CSV files.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-strategy MT5 historical research engine")
    parser.add_argument("--account-balance", type=float, default=10_000.0, help="research notional (never reads broker balance)")
    parser.add_argument("--risk-fraction", type=float, default=0.005)
    parser.add_argument("--spread-pips", type=float, default=None)
    parser.add_argument("--commission-pips-round-trip", type=float, default=None)
    parser.add_argument("--slippage-pips-round-trip", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "strategy-research")
    args = parser.parse_args()
    run_research(ResearchConfig(account_balance=args.account_balance, risk_fraction=args.risk_fraction,
                                spread_pips=args.spread_pips,
                                commission_pips_round_trip=args.commission_pips_round_trip,
                                slippage_pips_round_trip=args.slippage_pips_round_trip), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
