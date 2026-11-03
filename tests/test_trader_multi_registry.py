import json

import joblib
import numpy as np
import pandas as pd

from src.trader.multi_training import BASE_FEATURES
from src.trader.model_registry import (ModelRegistry, promote_multi_candidate,
                                       rollback_multi_model)


class FixedModel:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, values):
        return np.tile([1 - self.probability, self.probability], (len(values), 1))


def candles(count=1000, start="2026-01-01"):
    rng = np.random.default_rng(99)
    change = rng.normal(0.00003, 0.0001, count)
    close = 1.1 + np.cumsum(change)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"time": pd.date_range(start, periods=count, freq="5min", tz="UTC"),
                         "open": open_, "high": np.maximum(open_, close) + 0.00005,
                         "low": np.minimum(open_, close) - 0.00005, "close": close})


def test_multi_registry_uses_promoted_symbol_model_and_rolls_back(tmp_path, monkeypatch):
    import src.trader.multi_training as multi_training

    monkeypatch.setattr(multi_training, "_metrics", lambda frame, probability: {
        "rows": len(frame), "auc": 0.9, "brier_score": 0.1,
        "expected_calibration_error": 0.01, "trades": 100,
        "net_expectancy_pips": 3.0, "profit_factor": 1.5,
        "max_drawdown_return": -0.02, "precision": 0.8, "recall": 0.7,
        "directional_accuracy": 0.8,
    })
    current = tmp_path / "current.joblib"
    candidate = tmp_path / "eurusd-candidate.joblib"
    columns = [*BASE_FEATURES, "pair_EURUSD", *(f"timeframe_{x}" for x in
               ("M1", "M5", "M15", "H1", "H4"))]
    metadata = {"gate_passed": True, "model_kind": "pair", "symbol": "EURUSD",
                "timeframe": "M5", "horizon_bars": 3,
                "pair_universe": ["EURUSD"], "training_prevalence": 0.5,
                "holdout_end_utc": "2025-12-01T00:00:00+00:00"}
    joblib.dump({"status": "candidate", "model": FixedModel(0.8),
                 "feature_columns": columns, "metadata": metadata}, candidate)
    monkeypatch.setattr("src.trader.model_registry.brier_score_loss", lambda *args, **kwargs: 0.2)
    promoted = promote_multi_candidate(
        candidate, current, confirmation_candles={"EURUSD": candles()},
        round_trip_cost_pips=1.0, approved_by="test operator",
        minimum_sample_count=10, minimum_trades=10,
    )
    key = "pair:EURUSD:M5"
    assert promoted["model_key"] == key
    active_file = tmp_path / "active_models.json"
    assert json.loads(active_file.read_text())["active_models"][key]
    registry = ModelRegistry(current)
    assert registry.status["available"]
    probability = registry.predict_probability(
        {name: 0.0 for name in BASE_FEATURES}, timeframe="M5", symbol="EURUSD.a")
    assert probability == 0.8

    prior = tmp_path / "versions" / "prior.joblib"
    joblib.dump({"status": "promoted", "model": FixedModel(0.65),
                 "feature_columns": columns,
                 "metadata": {"version": "prior-version", "model_key": key,
                              "timeframe": "M5"}}, prior)
    assert rollback_multi_model(current, model_key=key, version="prior-version") == str(prior)
    assert ModelRegistry(current).predict_probability(
        {name: 0.0 for name in BASE_FEATURES}, timeframe="M5", symbol="EURUSD") == 0.65
