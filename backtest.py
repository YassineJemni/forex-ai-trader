
import pandas as pd
import numpy as np

FILE = "eurusd_clean.csv"

# Assumptions: adjust these to model your intended broker.
PIP = 0.0001
ROUND_TRIP_COST_PIPS = 1.0
MIN_TRADES = 100

# Strategy candidates to compare.
EMA_PAIRS = [(5, 21), (9, 21), (9, 50), (12, 50), (21, 50), (21, 100)]
RSI_THRESHOLDS = [50, 55, 60]

df = pd.read_csv(FILE)
df.columns = [str(c).lower() for c in df.columns]

df["time"] = pd.to_datetime(df["time"], errors="coerce")
for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=["time", "open", "close"])
    .sort_values("time")
    .drop_duplicates(subset=["time"])
    .reset_index(drop=True)
)

# Reject invalid prices.
df = df[
    (df["open"] > 0)
    & (df["close"] > 0)
].reset_index(drop=True)

# Signal is calculated at the current candle's close.
# Entry is at the NEXT candle's open; exit is at its close.
df["next_time"] = df["time"].shift(-1)
df["next_open"] = df["open"].shift(-1)
df["next_close"] = df["close"].shift(-1)

gap_minutes = (
    df["next_time"] - df["time"]
).dt.total_seconds() / 60

# Do not treat missing market minutes as consecutive candles.
valid_next = gap_minutes.between(0.5, 1.5)

df["future_pips"] = (
    (df["next_close"] - df["next_open"]) / PIP
)

# Three chronological periods:
# 60% search, 20% selection, 20% final untouched test.
n = len(df)
train_end = int(n * 0.60)
validation_end = int(n * 0.80)

# Calculate indicators causally from historical prices.
for period in sorted(set(x for pair in EMA_PAIRS for x in pair)):
    df[f"ema_{period}"] = df["close"].ewm(
        span=period, adjust=False
    ).mean()

delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi"] = 100 - (100 / (1 + rs))

df["valid"] = (
    valid_next
    & df["future_pips"].notna()
    & df["rsi"].notna()
)

def evaluate(start, end, fast, slow, threshold):
    part = df.iloc[start:end].copy()

    bullish = (
        (part[f"ema_{fast}"] > part[f"ema_{slow}"])
        & (part["rsi"] >= threshold)
    )
    bearish = (
        (part[f"ema_{fast}"] < part[f"ema_{slow}"])
        & (part["rsi"] <= 100 - threshold)
    )

    signal = np.select([bullish, bearish], [1, -1], default=0)

    mask = part["valid"].to_numpy() & (signal != 0)
    if not mask.any():
        return None

    directions = signal[mask]
    # Costs are deducted on every simulated trade.
    returns = (
        directions * part.loc[mask, "future_pips"].to_numpy()
        - ROUND_TRIP_COST_PIPS
    )

    return {
        "fast": fast,
        "slow": slow,
        "rsi_threshold": threshold,
        "trades": len(returns),
        "win_rate": float((returns > 0).mean()),
        "net_pips": float(returns.sum()),
        "avg_pips": float(returns.mean()),
        "profit_factor": (
            float(returns[returns > 0].sum()
                  / abs(returns[returns < 0].sum()))
            if (returns < 0).any()
            else float("inf")
        ),
    }

# Search candidate strategies on training data only.
candidates = []
for fast, slow in EMA_PAIRS:
    for threshold in RSI_THRESHOLDS:
        result = evaluate(0, train_end, fast, slow, threshold)
        if result and result["trades"] >= MIN_TRADES:
            candidates.append(result)

if not candidates:
    raise SystemExit(
        "No eligible strategies. Check the data and minimum trade count."
    )

candidates.sort(key=lambda x: x["avg_pips"], reverse=True)
shortlist = candidates[:10]

# Select among the shortlist using validation data.
validated = []
for candidate in shortlist:
    result = evaluate(
        train_end,
        validation_end,
        candidate["fast"],
        candidate["slow"],
        candidate["rsi_threshold"],
    )
    if result and result["trades"] >= MIN_TRADES:
        validated.append(result)

if not validated:
    raise SystemExit(
        "No candidates had enough validation trades. "
        "Try a lower MIN_TRADES value."
    )

validated.sort(key=lambda x: x["avg_pips"], reverse=True)
best = validated[0]

# Open this period only after selecting the strategy.
final_test = evaluate(
    validation_end,
    n,
    best["fast"],
    best["slow"],
    best["rsi_threshold"],
)

def show(label, result):
    print(f"\n--- {label} ---")
    if not result:
        print("Insufficient trades.")
        return
    for key, value in result.items():
        if key == "win_rate":
            print(f"{key}: {value:.2%}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

print("EUR/USD STRATEGY SEARCH")
print(f"Rows: {n:,}")
print(f"Assumed round-trip cost: {ROUND_TRIP_COST_PIPS:.2f} pips")
print(f"Candidates searched: {len(candidates)}")
print(f"Training period: {df['time'].iloc[0]} to "
      f"{df['time'].iloc[train_end - 1]}")
print(f"Validation period: {df['time'].iloc[train_end]} to "
      f"{df['time'].iloc[validation_end - 1]}")
print(f"Final test period: {df['time'].iloc[validation_end]} to "
      f"{df['time'].iloc[-1]}")

show("SELECTED STRATEGY — VALIDATION", best)
show("FINAL UNTOUCHED TEST", final_test)

print(
    "\nImportant: this is a research prototype, not a live-trading "
    "forecast. The cost is an assumption; actual spreads, slippage, "
    "execution, and strategy-selection bias can change the result."
)
