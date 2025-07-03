import pandas as pd
import sqlite3

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

for symbol in symbols:
    print(f"\n--- Truy vấn sentiment cho {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    query = """
    WITH price_context AS (
        SELECT 
            e."Earnings Date" AS earnings_date,
            e."EPS Estimate" AS eps_est,
            e."Reported EPS" AS eps_actual,
            (SELECT p1.close FROM price p1 
             WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
            (SELECT p2.close FROM price p2 
             WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT 
        pc.*,
        s.compound,
        ROUND((pc.close_after - pc.close_before) * 100.0 / pc.close_before, 2) AS price_change_pct,
        CASE 
            WHEN pc.eps_actual > pc.eps_est THEN 'Beat'
            WHEN pc.eps_actual < pc.eps_est THEN 'Miss'
            ELSE 'Meet'
        END AS eps_result
    FROM price_context pc
    LEFT JOIN sentiment s ON pc.earnings_date = s."Earnings Date"
    WHERE pc.close_before IS NOT NULL AND pc.close_after IS NOT NULL
    """
    df_sentiment_analysis = pd.read_sql_query(query, conn)
    print(df_sentiment_analysis.columns)
    print(df_sentiment_analysis.head())
    # Sau khi chắc chắn có cột 'compound' rồi mới chạy:
    if 'compound' in df_sentiment_analysis.columns and 'price_change_pct' in df_sentiment_analysis.columns:
        print(df_sentiment_analysis[['compound', 'price_change_pct']].corr())
    else:
        print("Không tìm thấy cột 'compound' hoặc 'price_change_pct'!")
    conn.close() 