import pandas as pd
import sqlite3

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

for symbol in symbols:
    print(f"\n--- Truy vấn cho {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    query = """
    WITH price_context AS (
        SELECT 
            e."Earnings Date" AS earnings_date,
            e."EPS Estimate" AS eps_est,
            e."Reported EPS" AS eps_actual,
            -- Giá 1 ngày trước earnings
            (SELECT p1.close FROM price p1 
             WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
            -- Giá 1 ngày sau earnings
            (SELECT p2.close FROM price p2 
             WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT * ,
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
    print(df_analysis)
    conn.close() 