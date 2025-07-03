import yfinance as yf
import pandas as pd
import numpy as np

def generate_sentiment(ticker_symbol, start_date, end_date, output_file):
    ticker = yf.Ticker(ticker_symbol)
    earnings_df = ticker.earnings_dates.reset_index()

    # Chuyển cột Earnings Date thành datetime và bỏ timezone
    earnings_df["Earnings Date"] = pd.to_datetime(earnings_df["Earnings Date"]).dt.tz_localize(None)
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    earnings_df = earnings_df[(earnings_df["Earnings Date"] >= start_date_dt) & (earnings_df["Earnings Date"] <= end_date_dt)]
    earnings_df['compound'] = np.random.uniform(-1, 1, size=len(earnings_df))
    sentiment_df = earnings_df[['Earnings Date', 'compound']]
    sentiment_df.to_csv(output_file, index=False)
    print(f"Đã tạo file sentiment giả định: {output_file}")

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2025-06-30"
    for symbol in symbols:
        output_file = f"{symbol}_sentiment.csv"
        generate_sentiment(symbol, start_date, end_date, output_file) 