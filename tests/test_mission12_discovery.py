import numpy as np
import pandas as pd

from src.research.mission12_discovery import (
    DiagonalGaussianHMM, _market_features, _select_threshold, metric_summary,
    build_paper_opportunities, paper_decision, purge_overlapping_labels,
    select_nonoverlapping,
)


class _NeverClassifier:
    def predict_proba(self, x):
        return np.tile([0.8, 0.2], (len(x), 1))


class _HighClassifier:
    def predict_proba(self, x):
        return np.tile([0.2, 0.8], (len(x), 1))


class _PositiveRegressor:
    def predict(self, x):
        return np.ones(len(x))


class _IdentityCalibrator:
    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-x[:, 0]))
        return np.column_stack([1 - p, p])


def test_hmm_forward_filter_is_causal_and_probabilities_sum_to_one():
    rng = np.random.default_rng(7)
    train = np.r_[rng.normal(-1, .2, (100, 3)), rng.normal(1, .2, (100, 3))]
    hmm = DiagonalGaussianHMM(states=3, iterations=4).fit(train)
    a = hmm.filter(np.array([[0., 0., 0.], [-.5, -.5, -.5], [0.8, 0.8, 0.8]]))
    b = hmm.filter(np.array([[0., 0., 0.], [-.5, -.5, -.5], [-50., -50., -50.]]))
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    assert not np.allclose(a[2], b[2])


def test_market_features_are_decision_time_indicators():
    n = 90
    close = pd.Series(np.linspace(1.0, 1.1, n))
    frame = pd.DataFrame({"segment": 0, "close": close, "high": close + .002,
                          "low": close - .002, "open": close - .0005,
                          "ema9": close - .0002, "ema21": close - .0004,
                          "ema50": close - .0008, "atr14": .002,
                          "rsi14": 55., "bb_lower": close - .003,
                          "bb_upper": close + .003, "momentum5": .001,
                          "body": .0005, "upper_wick": .001,
                          "lower_wick": .0005, "regime": "trending"})
    result = _market_features(frame)
    assert result.loc[n - 1, "return_1"] > 0
    assert result.loc[n - 1, "rsi14"] == .55
    assert result.loc[n - 1, "regime"] == "trending"


def test_threshold_requires_positive_expected_r_and_penalizes_small_samples():
    y = np.array([1, 1, 0, 0] * 40)
    p = np.array([.8, .7, .6, .2] * 40)
    r = np.array([1., .5, -1., -1.] * 40)
    expected = np.array([.1, -.1, .5, 1.] * 40)
    threshold, score, stats = _select_threshold(y, p, r, expected_net_r=expected, min_trades=20)
    assert .5 <= threshold <= .75
    assert stats["trades"] > 0
    assert score <= 1
    assert metric_summary(pd.DataFrame({"net_r": r}))['trades'] == 160


def test_overlapping_labels_are_reported_without_moving_rows_between_splits():
    rows = pd.DataFrame({"pair": ["EURUSD"] * 3, "timeframe": ["H1"] * 3,
                         "split": ["train", "train", "validation"],
                         "entry_timestamp": pd.to_datetime(["2020-01-01T00Z", "2020-01-01T01Z", "2020-01-02T00Z"]),
                         "exit_timestamp": pd.to_datetime(["2020-01-01T03Z", "2020-01-01T04Z", "2020-01-02T01Z"])})
    result, diagnostic = purge_overlapping_labels(rows)
    assert len(result) == 3
    assert result.split.tolist() == rows.split.tolist()
    assert diagnostic["within_split_overlapping_label_rows"] == 2
    assert diagnostic["cross_split_labels"] == 0


def test_event_weights_use_only_simultaneous_decision_time_signals():
    rows = pd.DataFrame({"pair": ["EURUSD", "EURUSD", "EURUSD"], "timeframe": ["H1"] * 3,
                         "split": ["train"] * 3, "strategy_id": ["a", "b", "c"],
                         "entry_timestamp": pd.to_datetime(["2020-01-01T00Z", "2020-01-01T00Z", "2020-01-01T01Z"]),
                         "exit_timestamp": pd.to_datetime(["2020-01-01T04Z", "2020-01-01T01Z", "2020-01-01T02Z"])})
    result, _ = purge_overlapping_labels(rows)
    assert result.sample_weight.iloc[0] == .5
    assert result.sample_weight.iloc[1] == .5
    assert result.sample_weight.iloc[2] == 1.0


def test_market_selector_picks_one_top_candidate_and_blocks_active_overlap():
    period = pd.DataFrame({"pair": ["EURUSD"] * 4, "timeframe": ["H1"] * 4,
                           "strategy_id": ["a", "b", "c", "d"],
                           "entry_timestamp": pd.to_datetime(["2020-01-01T00Z", "2020-01-01T00Z", "2020-01-01T01Z", "2020-01-01T04Z"]),
                           "exit_timestamp": pd.to_datetime(["2020-01-01T03Z", "2020-01-01T02Z", "2020-01-01T05Z", "2020-01-01T06Z"])})
    selected = select_nonoverlapping(period, np.array([.8, .9, .99, .7]), np.array([.5, .6, 1., .2]), .5)
    assert selected.tolist() == [False, True, False, True]


def test_paper_decision_fails_closed_for_unpromoted_model_and_nonpositive_expected_value():
    opportunities = pd.DataFrame([{"pair": "EURUSD", "timeframe": "H1", "strategy_id": "ema_crossover",
                                   "direction": "BUY", "entry_price": 1.1, "stop_loss": 1.09,
                                   "take_profit": 1.12, "feature": 1.0}])
    model = {"model_status": "research_only_not_promoted"}
    assert paper_decision(opportunities, model)["status"] == "no_trade"
    model.update({"classifier": _HighClassifier(),
                  "regressor": _PositiveRegressor(), "calibrator": _IdentityCalibrator(),
                  "threshold": .5, "feature_columns": ["feature"]})
    decision = paper_decision(opportunities, model)
    assert decision["status"] == "no_trade"
    assert decision["ranked_opportunities"][0]["probability_profitable"] > .5
    assert "model_is_research_only_and_not_promoted" in decision["ranked_opportunities"][0]["rejection_reasons"]
    model["model_status"] = "validated_for_research_paper_mode"
    decision = paper_decision(opportunities, model)
    assert decision["status"] == "opportunity"
    assert decision["decision"]["stop_loss"] == 1.09


def test_paper_decision_rejects_nonpositive_expected_r():
    opportunities = pd.DataFrame([{"pair": "GBPUSD", "timeframe": "M15", "strategy_id": "x",
                                   "direction": "SELL", "entry_price": 1.2, "stop_loss": 1.21,
                                   "take_profit": 1.18, "feature": 0.0}])
    class NegativeRegressor:
        def predict(self, x): return -0.05 * np.ones(len(x))
    model = {"model_status": "validated_for_research_paper_mode", "classifier": _NeverClassifier(),
             "regressor": NegativeRegressor(), "calibrator": _IdentityCalibrator(),
             "threshold": .5, "feature_columns": ["feature"]}
    result = paper_decision(opportunities, model)
    assert result["status"] == "no_trade"
    assert "expected_net_r_not_positive" in result["ranked_opportunities"][0]["rejection_reasons"]


def test_paper_opportunity_bridge_uses_closed_bars_and_fresh_utc_quote():
    now = pd.Timestamp("2026-01-10T10:00:00Z")
    n = 130
    times = pd.date_range(end=now - pd.Timedelta(hours=1), periods=n, freq="1h")
    close = np.full(n, 1.1)
    open_ = close.copy()
    high = close + .0001
    low = close - .0001
    open_[-1], close[-1], high[-1], low[-1] = 1.101, 1.105, 1.106, 1.101
    candles = pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close})
    model = {"include_hmm": False}
    quotes = {"EURUSD": {"time": now - pd.Timedelta(seconds=1), "bid": 1.1049, "ask": 1.1050}}
    candidates = build_paper_opportunities({("EURUSD", "H1"): candles}, quotes, model, now_utc=now)
    assert not candidates.empty
    assert "donchian_breakout" in set(candidates.strategy_id)
    assert candidates.entry_price.min() >= quotes["EURUSD"]["bid"]
    stale = {"EURUSD": {**quotes["EURUSD"], "time": now - pd.Timedelta(minutes=5)}}
    assert build_paper_opportunities({("EURUSD", "H1"): candles}, stale, model, now_utc=now).empty
