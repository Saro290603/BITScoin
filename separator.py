import pandas as pd

df = pd.read_csv("ADAUSDT_binance_closing_prices.csv")
df1 = df['close']
df1.to_csv('Dataset.csv')