from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

for symbol in symbols:
    print(f"\n--- Machine Learning cho {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    query = """
    WITH price_context AS (
        SELECT 
            e."Earnings Date" AS earnings_date,
            e."EPS Estimate" AS eps_est,
            e."Reported EPS" AS eps_actual,
            (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
            (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT *,
        ROUND((close_after - close_before) * 100.0 / close_before, 2) AS price_change_pct,
        CASE 
            WHEN eps_actual > eps_est THEN 'Beat'
            WHEN eps_actual < eps_est THEN 'Miss'
            ELSE 'Meet'
        END AS eps_result
    FROM price_context
    WHERE close_before IS NOT NULL AND close_after IS NOT NULL
    """
    df_analysis = pd.read_sql_query(query, conn)
    sentiment_df = pd.read_sql_query("SELECT * FROM sentiment", conn)
    conn.close()
    # Merge dữ liệu
    df_sentiment_analysis = pd.merge(df_analysis, sentiment_df, left_on='earnings_date', right_on='Earnings Date')
    df_sentiment_analysis['earnings_date'] = pd.to_datetime(df_sentiment_analysis['earnings_date'])
    # 1. Chuẩn bị dữ liệu
    X = df_sentiment_analysis[['compound']]
    y = df_sentiment_analysis['price_change_pct']
    # 2. Chia train-test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    # 4. Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    # 5. Kết quả
    print(f"RMSE Linear Regression: {rmse_lr:.2f}")
    print(f"RMSE Random Forest:     {rmse_rf:.2f}")
    if rmse_lr < 6 and rmse_rf < 6:
        print("✅ Cả hai mô hình đều đạt yêu cầu RMSE < 6%")
    else:
        print("⚠️ Có mô hình chưa đạt yêu cầu RMSE < 6%")
    # 6. Xuất dữ liệu phân tích cuối cùng để làm dashboard
    df_sentiment_analysis.reset_index(inplace=True)
    df_sentiment_analysis.to_csv(f'{symbol.lower()}_sentiment_analysis.csv', index=False) 