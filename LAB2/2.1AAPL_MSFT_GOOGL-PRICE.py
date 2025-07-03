import yfinance as yf
import pandas as pd

# Danh sách các mã cổ phiếu
symbols = ["AAPL", "MSFT", "GOOGL"]

# Tải và lưu dữ liệu giá cho từng mã
for symbol in symbols:
    price_data = yf.download(symbol, start='2023-01-01', end='2025-06-30')
    filename = f"{symbol}_price.csv"
    price_data.to_csv(filename)
    print(f"Đã lưu {filename}")