import requests
import pandas as pd

symbol = "ADAUSDT"
url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": symbol,
    "interval": "1d",
    "limit": 1000  # Binance allows fetching up to 1000 days at a time
}

response = requests.get(url, params=params)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df["close"] = df["close"].astype(float)

# Save to CSV
df.to_csv(f"{symbol}_binance_closing_prices.csv", index=False)
print(f"Saved {symbol} closing prices to CSV")
