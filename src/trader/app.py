from __future__ import annotations

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import AppConfig
from .history import ScanHistory
from .model_registry import ModelRegistry
from .mt5_adapter import MT5DataAdapter
from .scanner import ForexScanner


class AppState:
    def __init__(self, *, config: AppConfig | None = None,
                 adapter: MT5DataAdapter | None = None,
                 models: ModelRegistry | None = None,
                 history: ScanHistory | None = None):
        self.config = config or AppConfig.from_env()
        self.adapter = adapter or MT5DataAdapter(terminal_path=self.config.mt5_terminal_path,
                                                  max_quote_age_seconds=self.config.max_quote_age_seconds,
                                                  server_timezone=self.config.mt5_server_timezone)
        self.models = models or ModelRegistry(Path("models/current.joblib"))
        self.history = history or ScanHistory()
        self.scanner = ForexScanner(self.config, self.adapter, self.models,
                                    research_model=self.config.research_model_path)
        self.lock = threading.Lock()
        self.scan: dict[str, Any] = {"scan_id": None, "state": "idle", "completed": 0,
                                    "total": 0, "current": None, "result": None, "error": None}

    def start_fetch(self) -> str:
        with self.lock:
            if self.scan["state"] == "running":
                return self.scan["scan_id"]
            scan_id = uuid.uuid4().hex
            total = len(self.config.pairs) * len(self.config.timeframes)
            self.scan = {"scan_id": scan_id, "state": "running", "completed": 0,
                         "total": total, "current": None, "result": None, "error": None}
        threading.Thread(target=self._run_scan, args=(scan_id,), daemon=True).start()
        return scan_id

    def _run_scan(self, scan_id: str) -> None:
        def update(done: int, total: int, symbol: str, timeframe: str) -> None:
            with self.lock:
                if self.scan["scan_id"] == scan_id:
                    self.scan.update(completed=done, total=total,
                                     current={"symbol": symbol, "timeframe": timeframe})
        try:
            result = self.scanner.fetch(progress=update)
            result["scan_id"] = self.history.record_scan(result)
            state = "complete"
            error = None
        except Exception as exc:
            result = None
            state = "error"
            error = str(exc)
        with self.lock:
            if self.scan["scan_id"] == scan_id:
                self.scan.update(state=state, result=result, error=error,
                                 completed=self.scan["total"], current=None)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.scan)


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ForexAITrader/0.1"

        def _json(self, payload: Any, status: int = 200) -> None:
            raw = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                path = Path(__file__).with_name("dashboard.html")
                raw = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/api/status":
                self._json({**state.scanner.status(), "scan": state.snapshot()})
            elif self.path == "/api/scan":
                self._json(state.snapshot())
            elif self.path == "/api/history":
                self._json(state.history.recent())
            elif self.path == "/api/performance":
                self._json(state.history.performance())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/fetch":
                self._json({"scan_id": state.start_fetch(), "state": "running"}, 202)
            elif self.path.startswith("/api/outcomes/"):
                try:
                    prediction_id = int(self.path.rsplit("/", 1)[1])
                    size = int(self.headers.get("Content-Length", "0"))
                    if size > 8192:
                        raise ValueError("request body too large")
                    body = json.loads(self.rfile.read(size) or b"{}")
                    state.history.record_outcome(prediction_id, float(body["outcome_return"]))
                    self._json({"recorded": True})
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    self._json({"error": str(exc)}, 400)
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, format: str, *args: Any) -> None:
            # Keep local app logs concise and avoid printing request data.
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the read-only Forex AI Trader dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    state = AppState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Forex AI Trader dashboard: http://{args.host}:{args.port}")
    print("Read-only mode. No order submission is implemented.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.adapter.close()


if __name__ == "__main__":
    main()
