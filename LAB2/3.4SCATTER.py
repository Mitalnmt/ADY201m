import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

for symbol in symbols:
    print(f"\n--- Phân tích scatter & boxplot cho {symbol} ---")
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
    # Gộp dữ liệu sentiment với kết quả phân tích theo earnings_date
    df_sentiment_analysis = pd.merge(df_analysis, sentiment_df, left_on='earnings_date', right_on='Earnings Date')
    # Vẽ scatter plot
    sns.scatterplot(data=df_sentiment_analysis, x='compound', y='price_change_pct')
    plt.title(f'Sentiment Score vs. Price Change % ({symbol})')
    plt.xlabel('Sentiment Score (compound)')
    plt.ylabel('Price Change (%)')
    plt.show()
    # Thống kê mô tả
    print(df_sentiment_analysis[['compound', 'price_change_pct']].describe())
    # Boxplot theo eps_result
    sns.boxplot(x='eps_result', y='compound', data=df_sentiment_analysis)
    plt.title(f'Sentiment Score by EPS Result ({symbol})')
    plt.show()
    sns.boxplot(x='eps_result', y='price_change_pct', data=df_sentiment_analysis)
    plt.title(f'Price Change % by EPS Result ({symbol})')
    plt.show() 