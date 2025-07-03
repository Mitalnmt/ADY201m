import yfinance as yf
import pandas as pd
import numpy as np
import os

# Thiết lập tham số chung
TICKER_SYMBOL = "AAPL"
START_DATE = "2023-01-01"
END_DATE = "2025-05-31"

# Tạo thư mục lưu file nếu chưa có
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Tải dữ liệu giá cổ phiếu
def download_price_data():
    price_data = yf.download(TICKER_SYMBOL, start=START_DATE, end=END_DATE)
    price_data.to_csv(f"{DATA_DIR}/AAPL_price.csv")
    print("Đã lưu AAPL_price.csv")

# 2. Tải dữ liệu earnings
def download_earnings_data():
    ticker = yf.Ticker(TICKER_SYMBOL)
    earnings_df = ticker.earnings_dates.reset_index()
    earnings_df["Earnings Date"] = pd.to_datetime(earnings_df["Earnings Date"])
    earnings_df = earnings_df.set_index("Earnings Date")
    earnings_df = earnings_df.sort_index()
    earnings_df = earnings_df[START_DATE:END_DATE]
    earnings_df.to_csv(f"{DATA_DIR}/AAPL_earnings.csv")
    print("Đã lưu dữ liệu earnings vào file: AAPL_earnings.csv")

# 3. Tạo dữ liệu sentiment giả định
def generate_fake_sentiment():
    ticker = yf.Ticker(TICKER_SYMBOL)
    earnings_df = ticker.earnings_dates.reset_index()
    earnings_df["Earnings Date"] = pd.to_datetime(earnings_df["Earnings Date"]).dt.tz_localize(None)
    earnings_df = earnings_df[
        (earnings_df["Earnings Date"] >= pd.to_datetime(START_DATE)) &
        (earnings_df["Earnings Date"] <= pd.to_datetime(END_DATE))
    ]
    earnings_df['compound'] = np.random.uniform(-1, 1, size=len(earnings_df))
    sentiment_df = earnings_df[['Earnings Date', 'compound']]
    sentiment_df.to_csv(f"{DATA_DIR}/AAPL_sentiment.csv", index=False)
    print("Đã tạo file sentiment giả định: AAPL_sentiment.csv")

# Gọi các hàm
if __name__ == "__main__":
    download_price_data()
    download_earnings_data()
    generate_fake_sentiment()
    # 4. Sửa dòng header thứ 3 trong file AAPL_price.csv
    fixed_price_path = f"{DATA_DIR}/AAPL_price.csv"
    with open(fixed_price_path, "r") as f:
        lines = f.readlines()

    if len(lines) >= 3:
        lines[2] = "Date,Close,High,Low,Open,Volume\n"
        with open(fixed_price_path, "w") as f:
            f.writelines(lines)
        print("Đã cập nhật dòng header thứ 3 trong AAPL_price.csv")
    else:
        print("File AAPL_price.csv không đủ dòng để sửa header.")