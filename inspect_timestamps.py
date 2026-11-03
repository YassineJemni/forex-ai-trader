
import pandas as pd

df = pd.read_csv("eurusd_clean.csv", usecols=["time"])
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna().sort_values("time").drop_duplicates()

gaps = df["time"].diff().dt.total_seconds().div(60).dropna()

print("\n--- TIME INTERVALS ---")
print(gaps.value_counts().sort_index().head(20).to_string())

print("\n--- INTERVAL SUMMARY ---")
print(f"Median interval: {gaps.median():.2f} minutes")
print(f"Intervals <= 1.5 min: {(gaps <= 1.5).mean():.2%}")
print(f"Intervals > 1.5 min: {(gaps > 1.5).mean():.2%}")
print(f"Intervals > 5 min: {(gaps > 5).mean():.2%}")
print(f"Intervals > 60 min: {(gaps > 60).mean():.2%}")

print("\n--- LARGEST GAPS ---")
print(gaps.nlargest(10).to_string(index=False))
