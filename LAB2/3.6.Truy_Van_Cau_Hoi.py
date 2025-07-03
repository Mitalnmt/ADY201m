import sqlite3
import pandas as pd

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

for symbol in symbols:
    print(f"\n--- Truy vấn cho {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])

    # 1. Phân cụm cổ phiếu theo phản ứng giá sau earnings
    print("\n1. Phân cụm phản ứng giá:")
    query_cluster = '''
    WITH price_context AS (
        SELECT e."Earnings Date" AS earnings_date,
               (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
               (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT *,
        ROUND((close_after - close_before) * 100.0 / close_before, 2) AS price_change_pct,
        CASE
            WHEN ABS((close_after - close_before) * 100.0 / close_before) >= 5 THEN 'Strong'
            WHEN ABS((close_after - close_before) * 100.0 / close_before) >= 2 THEN 'Medium'
            ELSE 'Weak'
        END AS reaction_cluster
    FROM price_context
    WHERE close_before IS NOT NULL AND close_after IS NOT NULL
    '''
    print(pd.read_sql_query(query_cluster, conn))

    # 2. Chênh lệch giá 3 ngày trước/sau earnings
    print("\n2. Chênh lệch giá 3 ngày trước/sau earnings:")
    query_3d = '''
    SELECT e."Earnings Date",
           (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-3 day')) AS close_3_before,
           (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+3 day')) AS close_3_after,
           ROUND(((SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+3 day')) -
                  (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-3 day')))
                  * 100.0 /
                  (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-3 day')),
                  2) AS price_change_pct_3d
    FROM earnings e
    '''
    print(pd.read_sql_query(query_3d, conn))

    # 3. Window function: moving average 5 ngày
    print("\n3. Moving average 5 ngày:")
    query_ma = '''
    SELECT Date, close,
           AVG(close) OVER (ORDER BY Date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma_5
    FROM price
    '''
    print(pd.read_sql_query(query_ma + ' LIMIT 10', conn))

    # 4. JOIN earnings và sentiment
    print("\n4. JOIN earnings và sentiment:")
    query_join = '''
    SELECT e."Earnings Date", e."EPS Estimate", e."Reported EPS", s.compound
    FROM earnings e
    LEFT JOIN sentiment s ON e."Earnings Date" = s."Earnings Date"
    '''
    print(pd.read_sql_query(query_join, conn))

    # 5. Cổ phiếu có phản ứng giá mạnh nhất
    print("\n5. Cổ phiếu có phản ứng giá mạnh nhất:")
    query_strongest = '''
    WITH price_context AS (
        SELECT e."Earnings Date" AS earnings_date,
               (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
               (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT *,
        ROUND((close_after - close_before) * 100.0 / close_before, 2) AS price_change_pct
    FROM price_context
    WHERE close_before IS NOT NULL AND close_after IS NOT NULL
    ORDER BY ABS(price_change_pct) DESC
    LIMIT 1
    '''
    print(pd.read_sql_query(query_strongest, conn))

    # 6. Phân loại mức độ phản ứng giá
    print("\n6. Phân loại mức độ phản ứng giá:")
    query_classify = '''
    WITH price_reactions AS (
        SELECT e."Earnings Date",
               (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
               (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
        WHERE (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) IS NOT NULL
          AND (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) IS NOT NULL
    )
    SELECT 
        "Earnings Date",
        ROUND((close_after - close_before) * 100.0 / close_before, 2) AS price_change_pct,
        CASE 
            WHEN (close_after - close_before) * 100.0 / close_before >= 5 THEN 'Very Strong Positive'
            WHEN (close_after - close_before) * 100.0 / close_before >= 2 THEN 'Strong Positive'
            WHEN (close_after - close_before) * 100.0 / close_before >= 0.5 THEN 'Moderate Positive'
            WHEN (close_after - close_before) * 100.0 / close_before >= -0.5 THEN 'Neutral'
            WHEN (close_after - close_before) * 100.0 / close_before >= -2 THEN 'Moderate Negative'
            WHEN (close_after - close_before) * 100.0 / close_before >= -5 THEN 'Strong Negative'
            ELSE 'Very Strong Negative'
        END AS reaction_category
    FROM price_reactions
    ORDER BY price_change_pct DESC
    '''
    print(pd.read_sql_query(query_classify, conn))

    conn.close() 