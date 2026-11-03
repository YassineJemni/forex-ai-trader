"""Demonstrate causal 15-minute labeling and purged chronological splits.

Run from the repository root with:
    python -m src.research.demo_labels

Local Dukascopy M1 documentation describes candle start timestamps. The local
parquet timestamps are UTC-aware, and ``eurusd_clean.csv`` matches their values
with the timezone marker removed. The exact CSV export invocation is
unavailable. This demo uses the supported ``bar_open`` convention and
localizes the CSV's naive timestamp values to UTC. ``EXECUTION_DELAY`` remains
explicit and configurable.
"""

from .labels import build_labeled_samples, chronological_split


TIMESTAMP_CONVENTION = "bar_open"
EXECUTION_DELAY = "0s"  # Still requires a strictly later candle open.


def main() -> None:
    samples, counts = build_labeled_samples(
        horizon="15min",
        timestamp_convention=TIMESTAMP_CONVENTION,
        execution_delay=EXECUTION_DELAY,
        bar_duration="1min",
        gap_tolerance="90s",
    )
    splits = chronological_split(
        samples,
        train_fraction=0.60,
        validation_fraction=0.20,
        purge_interval="0min",
    )

    print("EUR/USD label generation (research demonstration only)")
    print(f"Timestamp convention: {TIMESTAMP_CONVENTION} (Dukascopy M1 candle start)")
    print(f"Execution delay: {EXECUTION_DELAY}")
    print("Times are UTC; exact CSV export invocation is unavailable.")
    print(f"Decision rows: {counts['total_decisions']:,}")
    print(f"Accepted samples: {counts['accepted']:,}")
    print(f"Rejected samples: {counts['rejected']:,}")
    print("Rejected by reason:")
    for reason, number in counts["rejected_by_reason"].items():
        print(f"  {reason}: {number:,}")
    print("Split sample counts:")
    for name, frame in splits.items():
        print(f"  {name}: {len(frame):,}")
    print("\nNo model is trained and no profitability is evaluated.")


if __name__ == "__main__":
    main()
