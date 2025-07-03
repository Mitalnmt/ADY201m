import pandas as pd
import sqlite3

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}
# Giả lập ngành cho từng mã
industry_map = {"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication"}

for symbol in symbols:
    print(f"\n--- Truy vấn nâng cao cho {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    # 1. Ngày volume bất thường (volume > mean + 2*std) - xử lý bằng pandas
    df_price = pd.read_sql_query('SELECT Date, [Unnamed: 5] as volume, close FROM price', conn)
    mean_vol = df_price['volume'].mean()
    std_vol = df_price['volume'].std()
    outlier_df = df_price[df_price['volume'] > mean_vol + 2*std_vol]
    print("Ngày volume bất thường:")
    print(outlier_df)
    # 2. Moving average 5 ngày cho giá đóng cửa
    query_ma = '''
    SELECT Date, close,
           AVG(close) OVER (ORDER BY Date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma_5
    FROM price
    '''
    df_ma = pd.read_sql_query(query_ma, conn)
    print("Moving average 5 ngày:")
    print(df_ma.head(10))
    # 3. Phân cụm theo ngành (giả lập)
    print(f"Ngành của {symbol}: {industry_map[symbol]}")
    conn.close() 