"""Explicit independent confirmation, model promotion, and rollback CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_PAIRS
from .model_registry import promote_multi_candidate, rollback_multi_model
from .multi_training import load_collected_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage explicitly validated model versions")
    commands = parser.add_subparsers(dest="command", required=True)
    promote = commands.add_parser("promote", help="confirm candidate on later collected data")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--data-dir", default="market-data/mt5-history")
    promote.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    promote.add_argument("--current", default="models/current.joblib")
    promote.add_argument("--round-trip-cost-pips", type=float, required=True)
    promote.add_argument("--approved-by", required=True)
    promote.add_argument("--minimum-samples", type=int, default=500)
    promote.add_argument("--minimum-trades", type=int, default=100)
    rollback = commands.add_parser("rollback", help="reactivate an archived promoted version")
    rollback.add_argument("--current", default="models/current.joblib")
    rollback.add_argument("--model-key", required=True)
    rollback.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "rollback":
        result = {"rolled_back_to": rollback_multi_model(
            args.current, model_key=args.model_key, version=args.version)}
    else:
        import joblib
        bundle = joblib.load(args.candidate)
        metadata = bundle.get("metadata", {})
        timeframe = metadata.get("timeframe")
        if not timeframe:
            raise SystemExit("candidate has no timeframe metadata")
        pairs = [x.strip().upper() for x in args.pairs.split(",") if x.strip()]
        frames = load_collected_frames(args.data_dir, pairs, timeframe)
        result = promote_multi_candidate(
            args.candidate, args.current, confirmation_candles=frames,
            round_trip_cost_pips=args.round_trip_cost_pips,
            approved_by=args.approved_by,
            minimum_sample_count=args.minimum_samples,
            minimum_trades=args.minimum_trades,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
