from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.trader.config import AppConfig
from src.trader.model_registry import ModelRegistry
from src.trader.mt5_adapter import Quote
from src.trader.scanner import ForexScanner


class MockAdapter:
    def __init__(self, candles):
        self.data = candles
        self.quote_calls = []

    def status(self):
        return {"connected": True, "read_only": True, "last_error": None}

    def quote(self, symbol):
        self.quote_calls.append(symbol)
        return Quote(symbol, 1.1, 1.1001, datetime.now(timezone.utc))

    def candles(self, symbol, timeframe, count):
        return synthetic_candles(timeframe=timeframe)

    def open_positions(self):
        return []


class PredictingRegistry:
    status = {"available": True, "version": "registry-summary", "reason": None}

    def __init__(self):
        self.calls = []

    def predict_probability(self, features, *, timeframe, symbol):
        self.calls.append((features.copy(), timeframe, symbol))
        return .7

    def selected_model_version(self, *, timeframe, symbol):
        return f"version-{symbol}-{timeframe}"


def synthetic_candles(n=400, timeframe="M1"):
    rng = np.random.default_rng(5)
    close = 1.1 + np.cumsum(rng.normal(0, .00008, n))
    minutes = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}[timeframe]
    step = pd.Timedelta(minutes=minutes)
    last_close = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    times = pd.date_range(end=last_close - step, periods=n, freq=step)
    return pd.DataFrame({"time": times, "open": close,
                         "high": close + .0001, "low": close - .0001,
                         "close": close, "volume": 10})


def test_fetch_scans_multiple_symbols_and_fails_closed_without_promoted_model(tmp_path):
    config = AppConfig(pairs=("EURUSD", "GBPUSD"), timeframes=("M1",), bars_per_scan=300)
    adapter = MockAdapter(synthetic_candles())
    registry = ModelRegistry(tmp_path / "models" / "current.joblib")
    scanner = ForexScanner(config, adapter, registry)
    result = scanner.fetch()
    assert result["status"] == "no_trade"
    assert result["decision"] is None
    assert len(result["observations"]) == 2
    assert all(x["probability"] is None and not x["eligible"] for x in result["observations"])
    assert set(adapter.quote_calls) == {"EURUSD", "GBPUSD"}
    assert "NO TRADE" in result["message"]


def test_quote_currency_conversion_finds_direct_or_inverse_pairs():
    config = AppConfig(pairs=("EURUSD.a", "USDJPYm"), timeframes=("M1",))
    scanner = ForexScanner(config, MockAdapter(synthetic_candles()), ModelRegistry(Path("missing.joblib")))
    quotes = {"EURUSD.a": Quote("EURUSD.a", 1.1, 1.1002, datetime.now(timezone.utc)),
              "USDJPYm": Quote("USDJPYm", 150, 150.01, datetime.now(timezone.utc))}
    assert scanner._quote_to_account_rate("USD", quotes) == 1
    assert scanner._quote_to_account_rate("JPY", quotes) == 1 / 150.01
    assert scanner._quote_to_account_rate("GBP", quotes) is None


def test_fetch_uses_promoted_model_interface_with_pair_context_and_risk_plan():
    config = AppConfig(pairs=("EURUSD", "GBPUSD"), timeframes=("M1",), bars_per_scan=300)
    models = PredictingRegistry()
    scanner = ForexScanner(config, MockAdapter(synthetic_candles()), models)
    result = scanner.fetch()
    assert result["status"] == "opportunity"
    assert result["decision"]["direction"] == "BUY"
    assert result["decision"]["model_version"].startswith("version-")
    assert result["decision"]["trade_plan"]["position_lots"] > 0
    assert all(call[1] == "M1" and call[2] in config.pairs for call in models.calls)
    assert all("peer_return_mean" in call[0] and "peer_return_dispersion" in call[0]
               for call in models.calls)


def test_fetch_can_attach_optional_research_paper_scan_without_changing_live_model():
    config = AppConfig(pairs=("EURUSD",), timeframes=("M1",), bars_per_scan=300)
    registry = ModelRegistry(Path("missing.joblib"))
    scanner = ForexScanner(config, MockAdapter(synthetic_candles()), registry,
                           research_model={"model_status": "research_only_not_promoted"})
    result = scanner.fetch()
    assert result["status"] == "no_trade"
    assert result["research_paper_scan"]["status"] == "no_trade"
    assert result["research_paper_scan"]["read_only"] is True
    assert result["research_paper_scan"]["live_trading"] is False


def test_fetch_attaches_all_strategy_market_intelligence_for_m15_h1_h4():
    config = AppConfig(pairs=("EURUSD",), timeframes=("M15", "H1", "H4"), bars_per_scan=300)
    scanner = ForexScanner(config, MockAdapter(synthetic_candles()), ModelRegistry(Path("missing.joblib")))
    result = scanner.fetch()
    report = result["market_intelligence"]
    assert report["mode"] == "research_paper_only"
    assert report["live_trading"] is False
    assert report["markets_evaluated"] == 3
    assert report["strategy_evaluations"] == 42
    assert report["evaluations_per_market"] == 14
    assert report["strategy_ids"]
    assert report["decision"] is None
