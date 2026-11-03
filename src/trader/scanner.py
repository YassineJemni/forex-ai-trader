from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig, TIMEFRAME_MINUTES
from .model_registry import ModelRegistry
from .mt5_adapter import MT5DataAdapter, MT5Error
from .risk import RiskSettings, build_trade_plan
from .multi_training import BASE_FEATURES, build_model_features


class ForexScanner:
    def __init__(self, config: AppConfig, adapter: MT5DataAdapter, models: ModelRegistry,
                 research_model: dict[str, Any] | str | None = None):
        self.config, self.adapter, self.models = config, adapter, models
        if isinstance(research_model, str):
            import joblib
            research_model = joblib.load(research_model)
        self.research_model = research_model

    def status(self) -> dict[str, Any]:
        research_status = ({"configured": False, "model_version": None, "mode": "disabled"}
                           if self.research_model is None else
                           {"configured": True, "model_version": self.research_model.get("model_version"),
                            "mode": "research_paper_only", "promoted": False})
        return {"market": self.adapter.status(), "model": self.models.status,
                "read_only": True, "pairs": list(self.config.pairs),
                "timeframes": list(self.config.timeframes), "research_model": research_status}

    def fetch(self, progress=None) -> dict[str, Any]:
        model_status = self.models.status
        observations: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        total = len(self.config.pairs) * len(self.config.timeframes)
        done = 0
        quote_cache = {}
        research_candles: dict[tuple[str, str], pd.DataFrame] = {}
        intelligence_candles: dict[tuple[str, str], pd.DataFrame] = {}
        exposure_clear: bool | None = None
        exposure_count: int | None = None
        try:
            open_positions = self.adapter.open_positions()
            exposure_count = len(open_positions)
            exposure_clear = exposure_count == 0
        except MT5Error as exc:
            errors.append({"symbol": "-", "timeframe": "-", "error": f"Could not verify portfolio exposure: {exc}"})
        exposure_error = ("Existing MT5 positions are open; aggregate exposure is not approved by research coordination."
                          if exposure_count else "Could not verify account exposure.")
        if model_status["available"]:
            exposure_safe = exposure_clear is True
        for symbol in self.config.pairs:
            try:
                quote_cache[symbol] = self.adapter.quote(symbol)
            except MT5Error as exc:
                errors.append({"symbol": symbol, "timeframe": "-", "error": str(exc)})
        # Collect the latest causal feature row for every configured market first,
        # so cross-pair context is computed from the same-time observations.
        feature_rows: list[dict[str, Any]] = []
        for symbol in self.config.pairs:
            quote = quote_cache.get(symbol)
            for timeframe in self.config.timeframes:
                done += 1
                if progress:
                    progress(done, total, symbol, timeframe)
                item: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe,
                                        "quote_time": quote.time.isoformat() if quote else None,
                                        "model_version": model_status.get("version"),
                                        "probability": None, "direction": None,
                                        "eligible": False, "reason": None}
                if quote is None:
                    item["reason"] = "Current quote unavailable or stale."
                    observations.append(item)
                    continue
                try:
                    candles = self.adapter.candles(symbol, timeframe, self.config.bars_per_scan)
                    compact_symbol = "".join(ch for ch in symbol.upper() if ch.isalpha())[:6]
                    if len(compact_symbol) == 6:
                        research_candles[(compact_symbol, timeframe)] = candles
                        if timeframe in ("M15", "H1", "H4"):
                            intelligence_candles[(compact_symbol, timeframe)] = candles
                    featured = build_model_features(candles, timeframe=timeframe)
                    required = [name for name in BASE_FEATURES if not name.startswith("peer_")]
                    ready = featured.dropna(subset=required)
                    if ready.empty:
                        raise MT5Error("Insufficient contiguous bars for indicator warm-up.")
                    row = ready.iloc[-1]
                    features = {name: float(row[name]) for name in required}
                    item.update({"price": float(row.close), "spread": quote.spread,
                                 "rsi_14": float(row.rsi_14), "ema_gap": float(row.ema_gap),
                                 "volatility_15": float(row.volatility_15),
                                 "candle_time": row.time.isoformat()})
                    feature_rows.append({"item": item, "features": features,
                                         "quote": quote, "symbol": symbol,
                                         "timeframe": timeframe, "row": row,
                                         "candles": candles})
                except (MT5Error, ValueError, KeyError, IndexError) as exc:
                    item["reason"] = str(exc)
                    errors.append({"symbol": symbol, "timeframe": timeframe, "error": str(exc)})
                    observations.append(item)

        for record in feature_rows:
            peers = [other["features"]["return_1"] for other in feature_rows
                     if other["timeframe"] == record["timeframe"]
                     and other["item"]["candle_time"] == record["item"]["candle_time"]
                     and other["symbol"] != record["symbol"]
                     and np.isfinite(other["features"]["return_1"])]
            record["features"]["peer_return_mean"] = float(np.mean(peers)) if peers else float("nan")
            record["features"]["peer_return_dispersion"] = float(np.std(peers)) if peers else float("nan")
            item, features = record["item"], record["features"]
            probability = self.models.predict_probability(
                features, timeframe=record["timeframe"], symbol=record["symbol"])
            item["model_version"] = self.models.selected_model_version(
                timeframe=record["timeframe"], symbol=record["symbol"])
            if probability is None:
                item["reason"] = (model_status.get("reason") or
                                   "No promoted model is validated for this pair/timeframe.")
            else:
                item["probability"] = probability
                item["features"] = {k: v for k, v in features.items() if np.isfinite(v)}
                item["direction"] = "BUY" if probability >= 0.55 else "SELL" if probability <= 0.45 else None
                if item["direction"] is None:
                    item["reason"] = "Model probability is inside the no-signal band (45%–55%)."
                else:
                    quote, symbol = record["quote"], record["symbol"]
                    atr = float(record["row"].atr_pct * record["row"].close)
                    stop_distance = max(atr * 1.5, quote.spread * 2)
                    compact_symbol = "".join(ch for ch in symbol[:6].upper() if ch.isalpha())
                    if len(compact_symbol) != 6:
                        item["reason"] = "Cannot identify base/quote currencies from the configured MT5 symbol."
                        observations.append(item)
                        continue
                    quote_ccy = compact_symbol[3:]
                    conversion = self._quote_to_account_rate(quote_ccy, quote_cache)
                    if not exposure_safe:
                        item["reason"] = exposure_error
                    elif conversion is None:
                        item["reason"] = f"No safe {quote_ccy}-to-{self.config.account_currency} conversion quote for sizing."
                    else:
                        plan = build_trade_plan(
                            direction=item["direction"],
                            entry=quote.ask if item["direction"] == "BUY" else quote.bid,
                            stop_distance=stop_distance, reward_multiple=2.0,
                            settings=RiskSettings(self.config.account_balance, self.config.risk_percent,
                                                  self.config.max_open_risk_percent,
                                                  self.config.account_currency),
                            quote_to_account_rate=conversion)
                        item["trade_plan"] = asdict(plan)
                        item["eligible"] = True
                        item["reason"] = "Promoted model threshold passed; confirm open exposure before acting."
            observations.append(item)
        eligible = [x for x in observations if x["eligible"]]
        eligible.sort(key=lambda x: abs(float(x["probability"]) - 0.5), reverse=True)
        observations.sort(key=lambda x: (not x["eligible"],
                                         -abs(float(x["probability"]) - 0.5)
                                         if x["probability"] is not None else 0.0))
        decision = eligible[0] if eligible else None
        confidence = (decision["probability"] if decision and decision["direction"] == "BUY"
                      else 1 - decision["probability"] if decision else None)
        message = (f"Selected {decision['symbol']} {decision['direction']} with "
                   f"{confidence:.1%} directional probability. {decision['reason']}"
                   if decision else
                   "NO TRADE: no opportunity met the active model and risk gates.")
        result = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                  "status": "opportunity" if eligible else "no_trade",
                  "decision": decision,
                  "ranked_opportunities": eligible,
                  "observations": observations, "errors": errors,
                  "model": model_status,
                  "message": message,
                  "read_only": True}
        try:
            from src.research.market_intelligence import coordinate_market_set
            canonical_quotes = {"".join(ch for ch in symbol.upper() if ch.isalpha())[:6]: quote
                                for symbol, quote in quote_cache.items()
                                if len("".join(ch for ch in symbol.upper() if ch.isalpha())[:6]) == 6}
            result["market_intelligence"] = coordinate_market_set(
                intelligence_candles, canonical_quotes,
                as_of_utc=datetime.now(timezone.utc),
                max_quote_age_seconds=self.config.max_quote_age_seconds,
                commission_pips_round_trip=self.config.commission_pips_round_trip,
                slippage_pips_round_trip=self.config.slippage_pips_round_trip,
                portfolio_exposure_clear=exposure_clear)
            result["market_intelligence"]["open_position_count"] = exposure_count
        except (ValueError, KeyError, TypeError) as exc:
            result["market_intelligence"] = {
                "mode": "research_paper_only", "read_only": True, "live_trading": False,
                "status": "no_trade", "decision": None, "ranked_opportunities": [],
                "reason": f"Market intelligence unavailable: {exc}",
            }
        if self.research_model is not None:
            try:
                from src.research.mission12_discovery import paper_scan
                research_quotes = {"".join(ch for ch in symbol.upper() if ch.isalpha())[:6]: quote
                                   for symbol, quote in quote_cache.items()}
                result["research_paper_scan"] = paper_scan(research_candles, research_quotes,
                                                           self.research_model)
            except (ValueError, KeyError, TypeError) as exc:
                result["research_paper_scan"] = {"status": "no_trade", "decision": None,
                                                  "ranked_opportunities": [], "read_only": True,
                                                  "live_trading": False,
                                                  "reason": f"Research paper scan unavailable: {exc}"}
        return result

    def _quote_to_account_rate(self, quote_currency: str, quotes: dict[str, Any]) -> float | None:
        account = self.config.account_currency
        if quote_currency == account:
            return 1.0
        for symbol, quote in quotes.items():
            pair = "".join(ch for ch in symbol[:6].upper() if ch.isalpha())
            if len(pair) != 6:
                continue
            if pair[:3] == quote_currency and pair[3:] == account:
                return quote.bid
            if pair[:3] == account and pair[3:] == quote_currency and quote.ask > 0:
                return 1.0 / quote.ask
        return None
