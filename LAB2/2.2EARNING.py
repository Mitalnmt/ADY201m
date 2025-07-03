import yfinance as yf
import pandas as pd

def download_earnings(ticker_symbol, start_date, end_date, output_file):
    ticker = yf.Ticker(ticker_symbol)
    earnings_df = ticker.earnings_dates
    earnings_df = earnings_df.reset_index()
    earnings_df["Earnings Date"] = pd.to_datetime(earnings_df["Earnings Date"])
    earnings_df = earnings_df.set_index("Earnings Date")
    earnings_df = earnings_df.sort_index()
    earnings_df = earnings_df[start_date:end_date]
    earnings_df.to_csv(output_file)
    print(f"Đã lưu dữ liệu earnings vào file: {output_file}")

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2025-06-30"
    for symbol in symbols:
        output_file = f"{symbol}_earnings.csv"
        download_earnings(symbol, start_date, end_date, output_file) 