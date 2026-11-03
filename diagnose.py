
import pandas as pd
import numpy as np

df = pd.read_csv("eurusd_1m.csv")
df.columns = [str(c).lower() for c in df.columns]
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna(subset=["close"]).sort_values("datetime")
df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

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

df["next_change"] = df["close"].shift(-1) - df["close"]
df["next_direction"] = np.sign(df["next_change"])

split = int(len(df) * 0.70)
test = df.iloc[split:].dropna(
    subset=["rsi", "next_direction"]
).copy()

trades = test[test["signal"] != 0].copy()
nonflat = trades[trades["next_direction"] != 0]
wins = (nonflat["signal"] == nonflat["next_direction"]).sum()

print("\n--- DATA DIAGNOSTIC ---")
print("Test candles:", len(test))
print("Signals:", len(trades))
print("Next candle up:", (trades["next_direction"] == 1).sum())
print("Next candle down:", (trades["next_direction"] == -1).sum())
print("Next candle flat:", (trades["next_direction"] == 0).sum())
print("Non-flat outcomes:", len(nonflat))

if len(nonflat):
    print("Accuracy excluding flat candles:", f"{wins / len(nonflat):.2%}")

print("\nSignal directions:")
print("UP signals:", (trades["signal"] == 1).sum())
print("DOWN signals:", (trades["signal"] == -1).sum())

print("\nUnique close prices:", df["close"].nunique())
print("Repeated consecutive closes:",
      (df["close"].diff() == 0).sum())
print("Median absolute candle movement:",
      df["close"].diff().abs().median())
