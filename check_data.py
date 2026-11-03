
import pandas as pd
from pathlib import Path

file = Path("eurusd_clean.csv")

if not file.exists():
    raise FileNotFoundError(f"Cannot find {file.resolve()}")

df = pd.read_csv(file)

print("\n--- CSV STRUCTURE ---")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head().to_string())

# Find likely price and timestamp columns
print("\n--- DATA QUALITY ---")
print("Missing values by column:")
print(df.isna().sum().to_string())

print("\nDuplicate rows:", df.duplicated().sum())

price_candidates = [
    c for c in df.columns
    if c.lower() in {"close", "bid_close", "close_bid"}
]

if price_candidates:
    col = price_candidates[0]
    prices = pd.to_numeric(df[col], errors="coerce").dropna()
    print("\nPrice column:", col)
    print("Unique prices:", prices.nunique())
    print("Repeated consecutive prices:", prices.diff().eq(0).sum())
    print("Median absolute price change:", prices.diff().abs().median())
else:
    print("\nNo obvious close column found yet.")
    print("We'll map the correct price columns from the header.")
