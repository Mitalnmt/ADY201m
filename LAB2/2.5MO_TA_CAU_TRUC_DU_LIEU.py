import pandas as pd

symbols = ["AAPL", "MSFT", "GOOGL"]

for symbol in symbols:
    print(f"\n--- Mô tả cấu trúc dữ liệu cho {symbol} ---")
    df = pd.read_csv(f'{symbol}_price.csv', skiprows=2)
    df.rename(columns={'Price': 'Adj Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print(df.info())
    print(df.describe()) 