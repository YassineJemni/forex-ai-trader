
import pandas as pd
import numpy as np

PIP = 0.0001
HORIZON_MINUTES = 15

df = pd.read_csv("eurusd_clean.csv", parse_dates=["time"])
df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

times = df["time"].to_numpy(dtype="datetime64[ns]")
opens = pd.to_numeric(df["open"], errors="coerce").to_numpy()
closes = pd.to_numeric(df["close"], errors="coerce").to_numpy()

rows = []

for i in range(len(df) - 1):
    entry = i + 1

    # Signal is available at candle i close; entry is next candle open.
    if (times[entry] - times[i]) / np.timedelta64(1, "m") > 1.5:
        continue

    exit_time = times[entry] + np.timedelta64(HORIZON_MINUTES, "m")
    exit_idx = np.searchsorted(times, exit_time, side="left")
    if exit_idx >= len(df):
        continue

    gaps = np.diff(times[i:exit_idx + 1]) / np.timedelta64(1, "m")
    if len(gaps) and np.any(gaps > 1.5):
        continue

    move = (closes[exit_idx] - opens[entry]) / PIP
    if move == 0:
        continue

    # File timestamps are treated as UTC; verify this with the data source.
    hour = pd.Timestamp(times[i]).hour

    if 0 <= hour < 7:
        session = "Asia (00-07 UTC)"
    elif 7 <= hour < 12:
        session = "London morning (07-12 UTC)"
    elif 12 <= hour < 16:
        session = "London/New York overlap (12-16 UTC)"
    else:
        session = "New York/late (16-24 UTC)"

    rows.append({
        "session": session,
        "hour": hour,
        "gross_pips": move,
    })

result = pd.DataFrame(rows)

print("\n--- EUR/USD SESSION ANALYSIS ---")
print("15-minute price movement; before trading costs.")
print(result.groupby("session", sort=False).agg(
    samples=("gross_pips", "size"),
    avg_gross_pips=("gross_pips", "mean"),
    positive_pct=("gross_pips", lambda x: (x > 0).mean() * 100),
    volatility_pips=("gross_pips", "std"),
).round(3).to_string())

print("\n--- BY UTC HOUR ---")
print(result.groupby("hour").agg(
    samples=("gross_pips", "size"),
    avg_gross_pips=("gross_pips", "mean"),
    positive_pct=("gross_pips", lambda x: (x > 0).mean() * 100),
).round(3).to_string())

print("\nExploratory only: session averages alone do not establish an edge.")
