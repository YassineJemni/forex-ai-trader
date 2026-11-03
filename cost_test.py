
import pandas as pd
import numpy as np

PIP = 0.0001
COSTS = [0.0, 0.5, 1.0, 1.5]
HORIZONS = [1, 5, 15, 60]

df = pd.read_csv("eurusd_clean.csv", parse_dates=["time"])
df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

for col in ["open", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "open", "close"]).reset_index(drop=True)

df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi"] = 100 - 100 / (1 + rs)

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

start = int(len(df) * 0.80)
rows = []

for horizon in HORIZONS:
    gross = []

    for i in range(start, len(df) - 1):
        if signals[i] == 0:
            continue

        entry = i + 1
        # Do not enter if the signal-to-entry interval has a gap.
        if (times[entry] - times[i]) / np.timedelta64(1, "m") > 1.5:
            continue

        target = times[entry] + np.timedelta64(horizon, "m")
        exit_idx = np.searchsorted(times, target, side="left")

        if exit_idx >= len(df):
            continue

        segment = times[i:exit_idx + 1]
        gaps = np.diff(segment) / np.timedelta64(1, "m")
        if np.any(gaps > 1.5):
            continue

        move = (
            (closes[exit_idx] - opens[entry]) / PIP
        ) * signals[i]
        gross.append(move)

    if not gross:
        continue

    gross = np.asarray(gross)
    for cost in COSTS:
        net = gross - cost
        rows.append({
            "horizon_min": horizon,
            "cost_pips": cost,
            "trades": len(net),
            "gross_avg_pips": gross.mean(),
            "gross_positive_pct": (gross > 0).mean() * 100,
            "net_avg_pips": net.mean(),
            "net_total_pips": net.sum(),
        })

result = pd.DataFrame(rows)
print("\n--- COST SENSITIVITY: FINAL 20% ---")
print(result.round(3).to_string(index=False))
