import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from src.trader.app import AppState, make_handler
from src.trader.config import AppConfig
from src.trader.history import ScanHistory
from src.trader.model_registry import ModelRegistry


def test_mt5_server_timezone_uses_verified_demo_default_and_allows_override(monkeypatch):
    monkeypatch.delenv("MT5_SERVER_TIMEZONE", raising=False)
    assert AppConfig.from_env().mt5_server_timezone == "Etc/GMT-2"
    monkeypatch.setenv("MT5_SERVER_TIMEZONE", "Europe/Budapest")
    assert AppConfig.from_env().mt5_server_timezone == "Europe/Budapest"


def test_research_paper_model_path_is_optional_and_env_configurable(monkeypatch):
    monkeypatch.delenv("FOREX_RESEARCH_MODEL_PATH", raising=False)
    assert AppConfig.from_env().research_model_path is None
    monkeypatch.setenv("FOREX_RESEARCH_MODEL_PATH", "reports/strategy-research/mission12/model.joblib")
    assert AppConfig.from_env().research_model_path.endswith("model.joblib")


def test_default_scan_timeframes_cover_market_intelligence_set(monkeypatch):
    monkeypatch.delenv("FOREX_TIMEFRAMES", raising=False)
    assert AppConfig.from_env().timeframes == ("M15", "H1", "H4")


def test_assumed_cost_inputs_are_optional_and_validated(monkeypatch):
    monkeypatch.setenv("FOREX_COMMISSION_PIPS_RT", "1.25")
    monkeypatch.setenv("FOREX_SLIPPAGE_PIPS_RT", "0.4")
    cfg = AppConfig.from_env()
    assert cfg.commission_pips_round_trip == 1.25
    assert cfg.slippage_pips_round_trip == 0.4
    monkeypatch.setenv("FOREX_COMMISSION_PIPS_RT", "-1")
    import pytest
    with pytest.raises(ValueError, match="nonnegative"):
        AppConfig.from_env()


class FakeAdapter:
    def status(self):
        return {"connected": False, "read_only": True, "last_error": None}

    def close(self):
        pass


def test_dashboard_api_fetch_and_history_end_to_end(tmp_path):
    state = AppState(config=AppConfig(pairs=("EURUSD",), timeframes=("M1",)),
                     adapter=FakeAdapter(), models=ModelRegistry(tmp_path / "current.joblib"),
                     history=ScanHistory(tmp_path / "history.sqlite3"))

    def fetch(progress=None):
        if progress:
            progress(1, 1, "EURUSD", "M1")
        return {"status": "no_trade", "message": "NO TRADE", "fetched_at": "2026-01-01T00:00:00+00:00",
                "decision": None, "ranked_opportunities": [],
                "observations": [{"symbol": "EURUSD", "timeframe": "M1", "probability": .51,
                                  "direction": None, "model_version": "m-test",
                                  "candle_time": "2026-01-01T00:00:00+00:00",
                                  "features": {"return_1": .01}}], "errors": [],
                "model": {"available": False}, "read_only": True}

    state.scanner.fetch = fetch
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/status") as response:
            status = json.load(response)
        assert status["read_only"] is True
        request = urllib.request.Request(base + "/api/fetch", data=b"{}", method="POST")
        with urllib.request.urlopen(request) as response:
            scan_id = json.load(response)["scan_id"]
        for _ in range(100):
            with urllib.request.urlopen(base + "/api/scan") as response:
                scan = json.load(response)
            if scan["state"] != "running":
                break
        assert scan["scan_id"] == scan_id
        assert scan["state"] == "complete"
        assert scan["result"]["status"] == "no_trade"
        with urllib.request.urlopen(base + "/api/history") as response:
            history = json.load(response)
        assert history["scans"][0]["status"] == "no_trade"
        prediction_id = history["predictions"][0]["prediction_id"]
        body = json.dumps({"outcome_return": 2.5}).encode()
        request = urllib.request.Request(base + f"/api/outcomes/{prediction_id}", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["recorded"] is True
        with urllib.request.urlopen(base + "/api/performance") as response:
            performance = json.load(response)
        assert performance["labeled_predictions"] == 1
    finally:
        server.shutdown()
        server.server_close()
