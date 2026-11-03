from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .history import ScanHistory
from .training import train_candidate_from_labeled_samples


def retrain_from_history(*, history_path: str | Path, symbol: str, timeframe: str,
                         output_path: str | Path, minimum_samples: int = 1000):
    """Train a candidate only from accumulated labeled observations; never promote."""
    samples = ScanHistory(history_path).labeled_samples(symbol=symbol, timeframe=timeframe)
    if len(samples) < minimum_samples:
        return {"status": "not_enough_labels", "labeled_samples": len(samples),
                "required": minimum_samples, "promoted": False}
    output_path = Path(output_path)
    if output_path.exists():
        existing = joblib.load(output_path)
        from .model_registry import _data_fingerprint
        import pandas as pd
        development = pd.DataFrame(samples).sort_values("time").reset_index(drop=True)
        development = development.iloc[:int(len(development) * .8)]
        if existing.get("metadata", {}).get("data_fingerprint") == _data_fingerprint(development):
            return {"status": "no_new_labeled_data", "labeled_samples": len(samples), "promoted": False}
    result = train_candidate_from_labeled_samples(samples, output_path, symbol=symbol,
                                                   timeframe=timeframe)
    return {"status": "candidate_created", "candidate": result.__dict__,
            "promoted": False, "note": "Candidate requires independent confirmation and explicit promotion."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an inert candidate from labeled prediction history")
    parser.add_argument("--history", default="results/trader_history.sqlite3")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--output", default="models/candidates/retrained.joblib")
    parser.add_argument("--minimum-samples", type=int, default=1000)
    args = parser.parse_args()
    result = retrain_from_history(history_path=args.history, symbol=args.symbol,
                                  timeframe=args.timeframe, output_path=args.output,
                                  minimum_samples=args.minimum_samples)
    print(result)


if __name__ == "__main__":
    main()
