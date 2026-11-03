from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import json
import os
import tempfile


@dataclass
class ModelRegistry:
    """Loads only explicitly promoted models; candidate artifacts are inert."""
    current_path: Path
    _bundle: dict[str, Any] | None = None
    _load_error: str | None = None
    _active_cache: list[dict[str, Any]] | None = None
    _active_mtime_ns: int | None = None

    @property
    def available(self) -> bool:
        return self._load() is not None or bool(self._load_active())

    @property
    def status(self) -> dict[str, Any]:
        active = self._load_active()
        legacy = self._load()
        if active:
            versions = [bundle["metadata"].get("version") for bundle in active]
            return {"available": True,
                    "version": versions[0] if len(versions) == 1 else f"{len(versions)} active model slots",
                    "reason": None,
                    "active_models": [{"model_key": bundle["metadata"].get("model_key"),
                                       "version": bundle["metadata"].get("version")}
                                      for bundle in active]}
        bundle = legacy
        if bundle is None:
            return {"available": False, "version": None,
                    "reason": (self._load_error or
                               "No promoted model is installed; no trading probability is available.")}
        return {"available": True, "version": bundle["metadata"]["version"], "reason": None}

    def _load(self) -> dict[str, Any] | None:
        if self._bundle is not None:
            return self._bundle
        if not self.current_path.exists():
            return None
        try:
            bundle = joblib.load(self.current_path)
        except Exception as exc:
            self._load_error = f"Promoted model artifact cannot be loaded: {exc}"
            return None
        if not isinstance(bundle, dict) or bundle.get("status") != "promoted":
            self._load_error = "Model artifact is not explicitly promoted."
            return None
        if not bundle.get("metadata", {}).get("version") or not bundle.get("feature_columns"):
            self._load_error = "Promoted model metadata is incomplete."
            return None
        self._bundle = bundle
        return bundle

    def _load_active(self) -> list[dict[str, Any]]:
        manifest_path = self.current_path.parent / "active_models.json"
        if not manifest_path.exists():
            self._active_cache = []
            self._active_mtime_ns = None
            return []
        mtime_ns = manifest_path.stat().st_mtime_ns
        if self._active_cache is not None and self._active_mtime_ns == mtime_ns:
            return self._active_cache
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            loaded = []
            for model_key, relative in manifest.get("active_models", {}).items():
                path = self.current_path.parent / relative
                bundle = joblib.load(path)
                if (bundle.get("status") != "promoted"
                        or bundle.get("metadata", {}).get("model_key") != model_key
                        or not bundle.get("feature_columns")):
                    raise ValueError(f"invalid promoted model entry: {model_key}")
                loaded.append(bundle)
            self._active_cache = loaded
            self._active_mtime_ns = mtime_ns
            return loaded
        except Exception as exc:
            self._load_error = f"Promoted model registry cannot be loaded: {exc}"
            return []

    def predict_probability(self, features: dict[str, float], *, timeframe: str,
                            symbol: str | None = None) -> float | None:
        bundle = self._select_bundle(timeframe, symbol)
        if bundle is None:
            return None
        columns = bundle["feature_columns"]
        compact = "".join(ch for ch in (symbol or "").upper() if ch.isalpha())[:6]
        row = dict(features)
        for name in columns:
            if name.startswith("pair_"):
                row[name] = float(name == f"pair_{compact}")
            elif name.startswith("timeframe_"):
                row[name] = float(name == f"timeframe_{timeframe}")
        if any(name not in row or np.isinf(row[name]) for name in columns):
            return None
        probability = float(bundle["model"].predict_proba([[row[n] for n in columns]])[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            return None
        return probability

    def _select_bundle(self, timeframe: str, symbol: str | None = None) -> dict[str, Any] | None:
        active = self._load_active()
        bundle = None
        compact = "".join(ch for ch in (symbol or "").upper() if ch.isalpha())[:6]
        if compact:
            specific = f"pair:{compact}:{timeframe}"
            bundle = next((x for x in active if x["metadata"].get("model_key") == specific), None)
        if bundle is None:
            shared = f"shared:ALL:{timeframe}"
            bundle = next((x for x in active if x["metadata"].get("model_key") == shared), None)
        if bundle is None:
            bundle = self._load()
        if bundle is None:
            return None
        if bundle["metadata"].get("timeframe") != timeframe:
            return None
        return bundle

    def selected_model_version(self, *, timeframe: str, symbol: str) -> str | None:
        bundle = self._select_bundle(timeframe, symbol)
        return bundle["metadata"].get("version") if bundle else None


def promote_multi_candidate(candidate_path: str | Path, current_path: str | Path, *,
                            confirmation_candles: dict[str, pd.DataFrame],
                            round_trip_cost_pips: float | dict[str, float],
                            approved_by: str, minimum_sample_count: int = 500,
                            minimum_trades: int = 100) -> dict[str, Any]:
    """Validate on later candles and explicitly promote one shared/pair slot."""
    if not approved_by.strip():
        raise ValueError("approved_by is required")
    candidate_path, current_path = Path(candidate_path), Path(current_path)
    bundle = joblib.load(candidate_path)
    if bundle.get("status") != "candidate" or not bundle.get("metadata", {}).get("gate_passed"):
        raise ValueError("candidate is invalid or failed its predeclared holdout gate")
    metadata = bundle["metadata"]
    if metadata.get("model_kind") not in {"shared", "pair"}:
        raise ValueError("candidate is not a multi-pair candidate")
    from .multi_training import build_multi_pair_samples, _metrics
    samples = build_multi_pair_samples(
        confirmation_candles, timeframe=metadata["timeframe"],
        horizon_bars=int(metadata["horizon_bars"]),
        round_trip_cost_pips=round_trip_cost_pips,
        pair_universe=metadata["pair_universe"],
    )
    samples = samples.loc[samples.time > pd.Timestamp(metadata["holdout_end_utc"])].copy()
    if metadata["model_kind"] == "pair":
        samples = samples.loc[samples.pair == metadata["symbol"]].copy()
    samples = samples.reset_index(drop=True)
    if len(samples) < minimum_sample_count:
        raise ValueError("independent post-holdout confirmation has too few labeled samples")
    columns = bundle["feature_columns"]
    probability = bundle["model"].predict_proba(samples[columns])[:, 1]
    metrics = _metrics(samples, probability)
    baseline = float(brier_score_loss(
        samples.target.astype(int),
        np.full(len(samples), metadata["training_prevalence"]),
    ))
    metrics["baseline_brier_score"] = baseline
    if (metrics["auc"] is None or metrics["auc"] <= .5
            or metrics["brier_score"] >= baseline
            or metrics["expected_calibration_error"] > .05
            or metrics["trades"] < minimum_trades
            or metrics["net_expectancy_pips"] <= 0):
        raise ValueError("independent confirmation failed predictive/calibration/cost gates")
    model_key = (f"shared:ALL:{metadata['timeframe']}" if metadata["model_kind"] == "shared"
                 else f"pair:{metadata['symbol']}:{metadata['timeframe']}")
    manifest_path = current_path.parent / "active_models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"active_models": {}}
    compare_keys = [model_key]
    if metadata["model_kind"] == "pair":
        compare_keys.append(f"shared:ALL:{metadata['timeframe']}")
    incumbent_results = {}
    for compare_key in compare_keys:
        incumbent_relative = manifest.get("active_models", {}).get(compare_key)
        if not incumbent_relative:
            continue
        incumbent = joblib.load(current_path.parent / incumbent_relative)
        incumbent_columns = incumbent["feature_columns"]
        if not set(incumbent_columns).issubset(samples.columns):
            raise ValueError(f"cannot compare candidate with {compare_key}: feature schema differs")
        old_probability = incumbent["model"].predict_proba(samples[incumbent_columns])[:, 1]
        old_metrics = _metrics(samples, old_probability)
        incumbent_results[compare_key] = {
            "brier_score": old_metrics["brier_score"],
            "net_expectancy_pips": old_metrics["net_expectancy_pips"],
        }
        if (metrics["brier_score"] >= old_metrics["brier_score"]
                or metrics["net_expectancy_pips"] <= old_metrics["net_expectancy_pips"]):
            raise ValueError(f"candidate failed to improve on promoted model {compare_key}")
    if incumbent_results:
        metrics["incumbents"] = incumbent_results
    from datetime import datetime, timezone
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    versions = current_path.parent / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    bundle["status"] = "promoted"
    bundle["metadata"].update({"version": version, "model_key": model_key,
                               "approved_by": approved_by.strip(),
                               "confirmation": metrics,
                               "confirmation_end_utc": samples.time.max().isoformat()})
    filename = f"{model_key.replace(':', '_')}_{version}.joblib"
    version_path = versions / filename
    joblib.dump(bundle, version_path)
    manifest.setdefault("active_models", {})[model_key] = str(version_path.relative_to(current_path.parent))
    _atomic_json(manifest, manifest_path)
    candidate_registry = current_path.parent / "candidate_registry.json"
    if candidate_registry.exists():
        entries = json.loads(candidate_registry.read_text(encoding="utf-8"))
        candidate_resolved = str(candidate_path.resolve())
        for entry in entries.get("candidates", []):
            if str(Path(entry.get("candidate_path", "")).resolve()) == candidate_resolved:
                entry["status"] = "promoted"
                entry["promoted_version"] = version
                entry["model_key"] = model_key
        _atomic_json(entries, candidate_registry)
    return bundle["metadata"]


def rollback_multi_model(current_path: str | Path, *, model_key: str, version: str) -> str:
    """Point one active model slot at an earlier archived, promoted version."""
    current_path = Path(current_path)
    versions_dir = current_path.parent / "versions"
    matches = []
    for path in versions_dir.glob("*.joblib"):
        try:
            bundle = joblib.load(path)
        except Exception:
            continue
        metadata = bundle.get("metadata", {})
        if (bundle.get("status") == "promoted" and metadata.get("model_key") == model_key
                and metadata.get("version") == version):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("requested promoted version was not found uniquely")
    manifest_path = current_path.parent / "active_models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"active_models": {}}
    manifest.setdefault("active_models", {})[model_key] = str(matches[0].relative_to(current_path.parent))
    _atomic_json(manifest, manifest_path)
    return str(matches[0])


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".registry-", suffix=".json", dir=path.parent)
    os.close(fd)
    try:
        Path(temp_name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def promote_candidate(candidate_path: str | Path, current_path: str | Path, *,
                      confirmation_candles, approved_by: str,
                      assumed_cost_pips: float = 1.0,
                      minimum_sample_count: int = 500,
                      minimum_trades: int = 100) -> dict[str, Any]:
    """Promote only after an explicit independent confirmation report passes.

    Metrics are computed in this function from caller-supplied independent
    candles. Their normalized input fingerprint and chronological boundaries
    must differ from and follow training data. The function atomically swaps
    the active artifact only after all gates pass.
    """
    if not approved_by.strip():
        raise ValueError("approved_by is required for a promotion record")
    candidate_path, current_path = Path(candidate_path), Path(current_path)
    bundle = joblib.load(candidate_path)
    if not isinstance(bundle, dict) or bundle.get("status") != "candidate":
        raise ValueError("candidate artifact is invalid")
    if not bundle.get("metadata", {}).get("gate_passed"):
        raise ValueError("candidate did not pass its predeclared development validation gate")
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from .training import FEATURES, make_samples, _score_non_overlapping, expected_calibration_error
    samples = make_samples(confirmation_candles, horizon_minutes=int(bundle["metadata"]["horizon_minutes"]),
                           round_trip_cost_pips=assumed_cost_pips)
    train_meta = bundle["metadata"]
    if len(samples) < minimum_sample_count:
        raise ValueError("independent confirmation has too few labeled samples")
    if train_meta.get("data_fingerprint") == _data_fingerprint(confirmation_candles):
        raise ValueError("confirmation data matches the candidate training fingerprint")
    development_end = pd.Timestamp(train_meta["development_end_utc"])
    if pd.Timestamp(samples.time.min()) <= development_end:
        raise ValueError("confirmation observations must occur after candidate train and validation data")
    x, y = samples[list(FEATURES)], samples.target.astype(int).to_numpy()
    probability = bundle["model"].predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, probability))
    brier = float(brier_score_loss(y, probability))
    calibration_error = expected_calibration_error(y, probability)
    baseline = float(brier_score_loss(y, np.full(len(y), train_meta["training_prevalence"])))
    evaluation = samples.copy()
    evaluation["Index"] = np.arange(len(evaluation))
    trades, expectancy = _score_non_overlapping(evaluation, probability, .55, assumed_cost_pips)
    confirmation = {"independent_data_fingerprint": _data_fingerprint(confirmation_candles),
                    "sample_count": len(samples), "auc": auc, "brier_score": brier,
                    "baseline_brier_score": baseline,
                    "expected_calibration_error": calibration_error,
                    "net_expectancy_after_assumed_costs": expectancy, "trades": trades,
                    "assumed_round_trip_cost_pips": assumed_cost_pips,
                    "minimum_sample_count": minimum_sample_count,
                    "minimum_trades": minimum_trades}
    if (auc <= .5 or brier >= baseline or calibration_error > .05
            or trades < minimum_trades or expectancy <= 0):
        raise ValueError("independent confirmation failed predictive or cost-adjusted gates")
    if current_path.exists():
        incumbent = joblib.load(current_path)
        if incumbent.get("status") == "promoted":
            if incumbent.get("metadata", {}).get("timeframe") != train_meta.get("timeframe"):
                raise ValueError("candidate and incumbent timeframes do not match")
            old_p = incumbent["model"].predict_proba(x)[:, 1]
            old_brier = float(brier_score_loss(y, old_p))
            _, old_expectancy = _score_non_overlapping(evaluation, old_p, .55, assumed_cost_pips)
            confirmation["incumbent_brier_score"] = old_brier
            confirmation["incumbent_net_expectancy"] = old_expectancy
            if brier >= old_brier or expectancy <= old_expectancy:
                raise ValueError("candidate did not improve on the active model")
    from datetime import datetime, timezone
    import os
    import shutil
    import tempfile
    current_path.parent.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_dir = current_path.parent / "versions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if current_path.exists():
        old_bundle = joblib.load(current_path)
        old_version = old_bundle.get("metadata", {}).get("version", "previous")
        old_archive = archive_dir / f"{old_version}.joblib"
        if not old_archive.exists():
            shutil.copy2(current_path, old_archive)
    bundle["status"] = "promoted"
    bundle["metadata"].update({"version": version, "approved_by": approved_by.strip(),
                               "confirmation": confirmation})
    fd, temp_name = tempfile.mkstemp(prefix="model-", suffix=".tmp", dir=current_path.parent)
    os.close(fd)
    promoted_archive = archive_dir / f"{version}.joblib"
    try:
        joblib.dump(bundle, temp_name)
        shutil.copy2(temp_name, promoted_archive)
        try:
            os.replace(temp_name, current_path)
        except Exception:
            promoted_archive.unlink(missing_ok=True)
            raise
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return bundle["metadata"]


def _data_fingerprint(data) -> str:
    import hashlib
    return hashlib.sha256(__import__("pandas").util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest()
