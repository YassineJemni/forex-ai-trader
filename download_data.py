
import yfinance as yf

symbol = "EURUSD=X"

print("Downloading EUR/USD historical data...")

data = yf.download(
    symbol,
    period="7d",
    interval="1m",
    auto_adjust=False,
    progress=False,
)

if data.empty:
    print("No data returned. We will try another data source.")
else:
    data.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in data.columns
    ]
    data = data.reset_index()
    data.columns = [str(col).lower().replace(" ", "_") for col in data.columns]
    data.to_csv("eurusd_1m.csv", index=False)

    print(f"Saved {len(data)} candles to eurusd_1m.csv")
    print(data.head())
