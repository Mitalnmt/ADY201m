import sqlite3
import pandas as pd

symbols = ["AAPL", "MSFT", "GOOGL"]

db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

for symbol in symbols:
    # Đọc CSV và chuyển đổi định dạng ngày tháng
    price_df = pd.read_csv(f'{symbol}_price.csv', header=2, parse_dates=['Date'])
    # Đổi tên cột giá đóng cửa thành 'close' nếu cần
    price_df.rename(columns={'Adj Close': 'close', 'Unnamed: 1': 'close'}, inplace=True)
    price_df['Date'] = price_df['Date'].dt.strftime('%Y-%m-%d')
    print(f"{symbol} price columns: {price_df.columns}")

    earnings_df = pd.read_csv(f'{symbol}_earnings.csv')
    earnings_df['Earnings Date'] = pd.to_datetime(earnings_df['Earnings Date'], errors='coerce', utc=True)
    earnings_df['Earnings Date'] = earnings_df['Earnings Date'].dt.tz_convert(None).dt.strftime('%Y-%m-%d')

    sentiment_df = pd.read_csv(f'{symbol}_sentiment.csv')
    sentiment_df['Earnings Date'] = pd.to_datetime(sentiment_df['Earnings Date'], errors='coerce', utc=True)
    sentiment_df['Earnings Date'] = sentiment_df['Earnings Date'].dt.tz_convert(None).dt.strftime('%Y-%m-%d')

    # Tạo kết nối DB SQLite riêng cho từng mã
    conn = sqlite3.connect(db_names[symbol])

    # Đưa dữ liệu vào DB
    price_df.to_sql('price', conn, if_exists='replace', index=False)
    earnings_df.to_sql('earnings', conn, if_exists='replace', index=False)
    sentiment_df.to_sql('sentiment', conn, if_exists='replace', index=False)
    print(f"Đã import dữ liệu cho {symbol} vào {db_names[symbol]}")

    # Đóng kết nối
    conn.close() 