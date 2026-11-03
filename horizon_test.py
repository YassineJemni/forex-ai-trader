
import pandas as pd
import numpy as np

FILE = "eurusd_clean.csv"
PIP = 0.0001
ROUND_TRIP_COST_PIPS = 1.0
HORIZONS = [1, 5, 15, 60]

df = pd.read_csv(FILE, parse_dates=["time"])
df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "open", "close"]).reset_index(drop=True)

# Indicators use only current and historical closing prices.
df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi"] = 100 - (100 / (1 + rs))

df["signal"] = np.select(
    [
        (df["ema9"] > df["ema21"]) & (df["rsi"] > 50),
        (df["ema9"] < df["ema21"]) & (df["rsi"] < 50),
    ],
    [1, -1],
    default=0,
)

times = df["time"].to_numpy(dtype="datetime64[ns]")
opens = df["open"].to_numpy()
closes = df["close"].to_numpy()
signals = df["signal"].to_numpy()

# Only use entries after a signal candle has closed.
# Exit at the first available candle close at or after each horizon.
results = []

for horizon in HORIZONS:
    returns = []
    test_start = int(len(df) * 0.80)

    for i in range(test_start, len(df) - 1):
        if signals[i] == 0:
            continue

        entry_idx = i + 1
        entry_time = times[entry_idx]
        target_time = entry_time + np.timedelta64(horizon, "m")

        # Find the first available candle at/after the target time.
        exit_idx = np.searchsorted(times, target_time, side="left")

        if exit_idx >= len(df):
            continue

        # Reject trades crossing a market/data gap.
        segment_times = times[entry_idx:exit_idx + 1]
        if len(segment_times) > 1:
            gaps = np.diff(segment_times) / np.timedelta64(1, "m")
            if np.any(gaps > 1.5):
                continue

        # Entry at next candle open, exit at target candle close.
        move_pips = (
            (closes[exit_idx] - opens[entry_idx]) / PIP
        ) * signals[i]

        net_pips = move_pips - ROUND_TRIP_COST_PIPS
        returns.append(net_pips)

    if returns:
        r = np.array(returns)
        results.append({
            "horizon_minutes": horizon,
            "trades": len(r),
            "win_rate": (r > 0).mean() * 100,
            "average_net_pips": r.mean(),
            "total_net_pips": r.sum(),
            "profit_factor": (
                r[r > 0].sum() / abs(r[r < 0].sum())
                if (r < 0).any() else np.inf
            ),
        })

result_df = pd.DataFrame(results)

print("\n--- EMA/RSI HORIZON EXPERIMENT ---")
print("Test uses the final 20% of the dataset.")
print(f"Assumed round-trip cost: {ROUND_TRIP_COST_PIPS:.1f} pip(s)")
print(result_df.round(3).to_string(index=False))
print("\nResearch experiment only; not a live-trading forecast.")
