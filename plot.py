import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("BTC-USD_5y_closing_prices.csv")

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"])
plt.show()