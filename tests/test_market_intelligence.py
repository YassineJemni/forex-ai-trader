from dataclasses import dataclass
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from src.research.market_intelligence import (
    STRATEGY_IDS,
    VALIDATION_SCENARIO,
    _independence_assessment,
    _validation_evidence_for,
    analyze_market,
    coordinate_market_set,
    load_pair_timeframe_validation,
)
from src.research.multi_strategy_engine import Strategy, build_features, strategy_library


AS_OF = pd.Timestamp("2026-03-01T12:00:00Z")
STEP_MINUTES = {"M15": 15, "H1": 60, "H4": 240}


@dataclass
class Quote:
    bid: float
    ask: float
    time: datetime = datetime(2026, 3, 1, 11, 59, 50, tzinfo=timezone.utc)

    @property
    def spread(self):
        return self.ask - self.bid


def candles(n=420, timeframe="M15", close_time=AS_OF):
    close = 1.05 + np.arange(n) * 0.00003 + np.sin(np.arange(n) / 5) * 0.00008
    step = pd.Timedelta(minutes=STEP_MINUTES[timeframe])
    return pd.DataFrame({
        "time": pd.date_range(end=close_time - step, periods=n, freq=step),
        "open": close - 0.00001, "high": close + 0.00008,
        "low": close - 0.00008, "close": close, "volume": 10,
    })


def breakout_candles(timeframe="M15", close_time=AS_OF):
    frame = candles(timeframe=timeframe, close_time=close_time)
    breakout = float(frame.high.iloc[-21:-1].max()) + 0.0004
    frame.loc[frame.index[-1], "open"] = breakout - 0.0002
    frame.loc[frame.index[-1], "close"] = breakout
    frame.loc[frame.index[-1], "high"] = breakout + 0.0001
    frame.loc[frame.index[-1], "low"] = breakout - 0.0003
    return frame


def coordinate(markets, *, quotes=None, **kwargs):
    return coordinate_market_set(
        markets, quotes or {"EURUSD": Quote(1.06, 1.0601)},
        as_of_utc=AS_OF, pair_timeframe_evidence={}, validation_correlations={}, **kwargs)


def test_market_intelligence_is_causal_and_uses_closed_candle_features():
    frame = candles()
    prior = analyze_market(build_features(frame.iloc[:350], "M15"), "M15")
    changed_future = frame.copy()
    changed_future.loc[350:, ["open", "high", "low", "close"]] *= 1.7
    same_prefix = analyze_market(build_features(changed_future.iloc[:350], "M15"), "M15")
    assert prior == same_prefix
    assert prior["support_20"] <= prior["resistance_20"]
    assert prior["session_utc"].startswith(("Asia", "London", "NewYork", "Rollover"))
    assert "potential_change_of_character" in prior


def test_all_fourteen_strategies_get_records_including_no_signal(monkeypatch):
    import src.research.market_intelligence as intelligence

    strategies = strategy_library()
    monkeypatch_values = tuple(
        Strategy(strategy.strategy_id, strategy.family, strategy.explanation,
                 lambda frame: pd.Series(0, index=frame.index))
        for strategy in strategies
    )
    monkeypatch.setattr(intelligence, "strategy_library", lambda: monkeypatch_values)
    result = coordinate({("EURUSD", "M15"): candles()})
    records = result["strategy_evaluation_records"]
    assert result["strategy_ids"] == list(STRATEGY_IDS)
    assert len(records) == 14
    assert {row["strategy_id"] for row in records} == set(STRATEGY_IDS)
    assert all(row["direction"] == "NO SIGNAL" for row in records)
    assert all(row["evaluation_status"] == "evaluated" for row in records)
    assert all(row["decision_timestamp"] == AS_OF.isoformat() for row in records)


def test_strategy_exceptions_are_recorded_without_dropping_other_strategies(monkeypatch):
    import src.research.market_intelligence as intelligence

    def fail(frame):
        if frame.empty:
            raise AssertionError("features unexpectedly empty")
        raise RuntimeError("synthetic strategy failure")

    strategies = list(strategy_library())
    strategies[0] = Strategy(
        strategies[0].strategy_id, strategies[0].family, strategies[0].explanation,
        fail)
    monkeypatch.setattr(intelligence, "strategy_library", lambda: tuple(strategies))
    result = coordinate({("EURUSD", "M15"): candles()})
    errors = result["strategy_evaluation_records"]
    assert len(errors) == 14
    failed = next(row for row in errors if row["strategy_id"] == strategies[0].strategy_id)
    assert failed["evaluation_status"] == "error"
    assert failed["error"] == "synthetic strategy failure"


def test_future_candle_and_future_quote_are_rejected_at_shared_cutoff():
    future = candles()
    future.loc[future.index[-1], "time"] += pd.Timedelta(minutes=1)
    result = coordinate({("EURUSD", "M15"): future})
    assert "future or not-yet-closed" in result["market_rejections"][0]["reason"]

    future_quote = Quote(1.06, 1.0601, time=(AS_OF + pd.Timedelta(seconds=1)).to_pydatetime())
    result = coordinate({("EURUSD", "M15"): candles()}, quotes={"EURUSD": future_quote})
    assert "after the shared as-of cutoff" in result["market_rejections"][0]["reason"]


def test_empty_candles_and_invalid_quotes_are_rejected_with_reasons():
    empty = candles().iloc[0:0]
    result = coordinate({("EURUSD", "M15"): empty})
    assert result["market_rejections"][0]["reason"] == "no candles supplied"

    crossed = Quote(1.0602, 1.0601)
    result = coordinate({("EURUSD", "M15"): candles()}, quotes={"EURUSD": crossed})
    assert result["market_rejections"][0]["reason"] == "invalid bid/ask quote values"


def test_stale_candle_and_stale_quote_are_rejected():
    old_close = AS_OF - pd.Timedelta(minutes=20)
    result = coordinate({("EURUSD", "M15"): candles(close_time=old_close)})
    assert "stale M15 candles" in result["market_rejections"][0]["reason"]

    stale_quote = Quote(1.06, 1.0601, time=(AS_OF - pd.Timedelta(minutes=2)).to_pydatetime())
    result = coordinate({("EURUSD", "M15"): candles()}, quotes={"EURUSD": stale_quote})
    assert "stale quote" in result["market_rejections"][0]["reason"]


def test_all_timeframes_share_one_cutoff_and_alignment_uses_only_closed_candles():
    markets = {( "EURUSD", timeframe): breakout_candles(timeframe)
               for timeframe in ("M15", "H1", "H4")}
    result = coordinate(markets, commission_pips_round_trip=1.0,
                        slippage_pips_round_trip=0.5, portfolio_exposure_clear=True)
    assert result["as_of_utc"] == AS_OF.isoformat()
    assert result["markets_evaluated"] == 3
    assert result["market_evaluation_counts"] == {
        "EURUSD_H4": 14, "EURUSD_H1": 14, "EURUSD_M15": 14}
    for signal in result["signals"]:
        signal_time = pd.Timestamp(signal["decision_timestamp"])
        assert all(pd.Timestamp(value) <= signal_time
                   for value in signal["multi_timeframe_close_times"].values())


def test_missing_higher_timeframe_never_counts_as_confirmation(monkeypatch):
    import src.research.market_intelligence as intelligence

    strategies = list(strategy_library())
    first = strategies[0]
    strategies[0] = Strategy(first.strategy_id, first.family, first.explanation,
                             lambda frame: pd.Series(1, index=frame.index))
    monkeypatch.setattr(intelligence, "strategy_library", lambda: tuple(strategies))
    result = coordinate({("EURUSD", "M15"): candles()})
    signal = result["signals"][0]
    assert signal["missing_timeframes"] == ["H1", "H4"]
    assert signal["multi_timeframe_coverage"] == ["M15"]
    assert signal["multi_timeframe_trend_alignment_count"] <= 1


def test_future_higher_timeframe_is_rejected_and_excluded_from_alignment(monkeypatch):
    import src.research.market_intelligence as intelligence

    strategies = list(strategy_library())
    first = strategies[0]
    strategies[0] = Strategy(first.strategy_id, first.family, first.explanation,
                             lambda frame: pd.Series(1, index=frame.index))
    monkeypatch.setattr(intelligence, "strategy_library", lambda: tuple(strategies))
    future_h1 = candles(timeframe="H1", close_time=AS_OF + pd.Timedelta(hours=1))
    result = coordinate({("EURUSD", "M15"): candles(),
                         ("EURUSD", "H1"): future_h1})
    assert any(row["timeframe"] == "H1" and "future" in row["reason"]
               for row in result["market_rejections"])
    signal = result["signals"][0]
    assert "H1" in signal["missing_timeframes"]
    assert "H1" not in signal["multi_timeframe_coverage"]


def test_validation_evidence_requires_exact_scope_stability_sample_and_positive_net():
    key = ("breakout", "EURUSD", "M15")
    base = {"strategy_id": "breakout", "pair": "EURUSD", "timeframe": "M15",
            "split": "validation", "scenario": VALIDATION_SCENARIO,
            "pair_timeframe_stability_status": "stable_pair_timeframe_validation",
            "trades": 25, "expectancy_r": 0.1}
    evidence = {key: base}
    selected, reasons, scope = _validation_evidence_for(
        "breakout", "EURUSD", "M15", "trending", evidence)
    assert selected == base
    assert reasons == []
    assert scope == "pair_timeframe"

    too_few = {key: {**base, "trades": 24}}
    selected, reasons, _ = _validation_evidence_for(
        "breakout", "EURUSD", "M15", "trending", too_few)
    assert selected is None
    assert any("insufficient validation sample" in reason for reason in reasons)

    pooled = {("breakout", "ALL", "ALL"): base}
    selected, reasons, _ = _validation_evidence_for(
        "breakout", "EURUSD", "M15", "trending", pooled)
    assert selected is None
    assert "no validation evidence for this exact strategy/pair/timeframe" in reasons

    unstable = {key: {**base, "pair_timeframe_stability_status": "unknown"}}
    selected, reasons, _ = _validation_evidence_for(
        "breakout", "EURUSD", "M15", "trending", unstable)
    assert selected is None
    assert any("stability is not established" in reason for reason in reasons)

    test_split = {key: {**base, "split": "test"}}
    selected, reasons, _ = _validation_evidence_for(
        "breakout", "EURUSD", "M15", "trending", test_split)
    assert selected is None
    assert "validation evidence scope or split does not match this market" in reasons


def test_mission9_loader_filters_validation_rows_and_does_not_promote_pooled_status(tmp_path):
    report = {
        "candidate_assessment_validation_only": [{
            "strategy_id": "breakout",
            "descriptive_status": "validation_candidate_for_more_evidence",
        }],
        "metrics": [
            {"strategy_id": "breakout", "pair": "EURUSD", "timeframe": "M15",
             "split": "validation", "scenario": VALIDATION_SCENARIO, "trades": 30},
            {"strategy_id": "breakout", "pair": "EURUSD", "timeframe": "M15",
             "split": "test", "scenario": VALIDATION_SCENARIO, "trades": 1000},
        ],
        "market_regime": [],
    }
    path = tmp_path / "mission9.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = load_pair_timeframe_validation(path)
    row = loaded[("breakout", "EURUSD", "M15")]
    assert row["split"] == "validation"
    assert row["pair_timeframe_stability_status"] == "unstable_or_insufficient_validation"


def test_unknown_strategy_correlations_award_no_independence_bonus():
    bonus, independent, correlated, unknown = _independence_assessment(
        ["strategy_a", "strategy_b"], {})
    assert bonus == 0
    assert independent == []
    assert correlated == []
    assert unknown == ["strategy_b"]

    bonus, independent, correlated, unknown = _independence_assessment(
        ["strategy_a", "strategy_b"], {("strategy_a", "strategy_b"): 0.2})
    assert bonus == 0.75
    assert independent == ["strategy_b"]
    assert correlated == []
    assert unknown == []


def test_unknown_independence_is_reported_without_coordination_bonus(monkeypatch):
    import src.research.market_intelligence as intelligence

    first = Strategy("strategy_a", "family_a", "first",
                     lambda frame: pd.Series(1, index=frame.index))
    second = Strategy("strategy_b", "family_b", "second",
                      lambda frame: pd.Series(1, index=frame.index))
    monkeypatch.setattr(intelligence, "strategy_library", lambda: (first, second))
    result = coordinate({("EURUSD", "M15"): candles()})
    opportunity = result["ranked_opportunities"][0]
    assert opportunity["independent_confirmations"] == []
    assert opportunity["unknown_independence_confirmations"] == ["strategy_b"]
    assert "no independence credit" in opportunity["independence_uncertainty"]
    assert opportunity["coordination_score"] == 1.0


def test_validation_and_test_splits_are_not_mixed():
    evidence = {
        ("strategy_a", "EURUSD", "M15"): {
            "strategy_id": "strategy_a", "pair": "EURUSD", "timeframe": "M15",
            "split": "validation", "scenario": VALIDATION_SCENARIO,
            "pair_timeframe_stability_status": "stable_pair_timeframe_validation",
            "trades": 40, "expectancy_r": 0.2,
            "regime_metrics": {"trending": {
                "strategy_id": "strategy_a", "pair": "EURUSD", "timeframe": "M15",
                "split": "test", "scenario": VALIDATION_SCENARIO,
                "trades": 10000, "expectancy_r": 5.0,
            }},
        }
    }
    selected, reasons, _ = _validation_evidence_for(
        "strategy_a", "EURUSD", "M15", "trending", evidence)
    assert selected is None
    assert any("scope or split" in reason for reason in reasons)


def test_invalid_historical_h4_rows_are_excluded_and_quarantined_without_changing_sources(tmp_path):
    from src.trader.multi_training import load_collected_frames

    source = pd.DataFrame({
        "time": pd.to_datetime(["1998-12-01T00:00:00Z", "1999-01-01T00:00:00Z"]),
        "open": [1.0, 1.1], "high": [1.2, 1.3],
        "low": [0.9, 1.0], "close": [1.1, 1.2],
    })
    for pair in ("EURUSD", "USDJPY"):
        source.to_parquet(tmp_path / f"{pair}_H4.parquet", index=False)
    original_eur = pd.read_parquet(tmp_path / "EURUSD_H4.parquet")
    original_jpy = pd.read_parquet(tmp_path / "USDJPY_H4.parquet")

    frames = load_collected_frames(tmp_path, ["EURUSD", "USDJPY"], "H4")
    assert all(frame.time.min() >= pd.Timestamp("1999-01-01T00:00:00Z")
               for frame in frames.values())
    assert len(frames["EURUSD"]) == 1
    assert len(frames["USDJPY"]) == 1
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "EURUSD_H4.parquet"), original_eur)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "USDJPY_H4.parquet"), original_jpy)
    assert (tmp_path / "quarantine" / "USDJPY_H4_pre1999.parquet").exists()
    assert (tmp_path / "quarantine" / "USDJPY_H4_pre1999.provenance.json").exists()
    assert (tmp_path / "derived-research" / "EURUSD_H4.provenance.json").exists()
    assert (tmp_path / "derived-research" / "USDJPY_H4.provenance.json").exists()
