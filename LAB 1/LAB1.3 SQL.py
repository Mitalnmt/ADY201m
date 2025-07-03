import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. IMPORT DỮ LIỆU VÀO SQLITE DB ===
data_folder = "data"
price_df = pd.read_csv(os.path.join(data_folder, 'AAPL_price.csv'), header=2, parse_dates=['Date'])
price_df['Date'] = price_df['Date'].dt.strftime('%Y-%m-%d')

earnings_df = pd.read_csv(os.path.join(data_folder, 'AAPL_earnings.csv'))
earnings_df['Earnings Date'] = pd.to_datetime(earnings_df['Earnings Date'], errors='coerce', utc=True)
earnings_df['Earnings Date'] = earnings_df['Earnings Date'].dt.tz_convert(None).dt.strftime('%Y-%m-%d')

sentiment_df = pd.read_csv(os.path.join(data_folder, 'AAPL_sentiment.csv'))
sentiment_df['Earnings Date'] = pd.to_datetime(sentiment_df['Earnings Date'], errors='coerce', utc=True)
sentiment_df['Earnings Date'] = sentiment_df['Earnings Date'].dt.tz_convert(None).dt.strftime('%Y-%m-%d')

# Tạo DB
conn = sqlite3.connect('aapl_analysis.db')
price_df.to_sql('price', conn, if_exists='replace', index=False)
earnings_df.to_sql('earnings', conn, if_exists='replace', index=False)
sentiment_df.to_sql('sentiment', conn, if_exists='replace', index=False)

# === 2. PHÂN TÍCH EPS VÀ GIÁ ===
query = """
WITH price_context AS (
    SELECT 
        e."Earnings Date" AS earnings_date,
        e."EPS Estimate" AS eps_est,
        e."Reported EPS" AS eps_actual,
        (SELECT p1.Close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
        (SELECT p2.Close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
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
sentiment_db = pd.read_sql_query("SELECT * FROM sentiment", conn)
conn.close()

df_sentiment_analysis = pd.merge(df_analysis, sentiment_db, left_on='earnings_date', right_on='Earnings Date')

# === 3. TẠO THƯ MỤC LƯU BIỂU ĐỒ ===
chart_dir = os.path.join("chart", "3 SQL")
os.makedirs(chart_dir, exist_ok=True)

# === 4. HIỂN THỊ & LƯU BIỂU ĐỒ ===
sns.set(style="whitegrid")

# Scatter: compound vs price_change_pct
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_sentiment_analysis, x='compound', y='price_change_pct')
plt.title('Sentiment Score vs. Price Change %')
plt.xlabel('Sentiment Score (compound)')
plt.ylabel('Price Change (%)')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'scatter_sentiment_vs_price_change.png'))
plt.show()

# Boxplot: compound theo eps_result
plt.figure(figsize=(8, 6))
sns.boxplot(x='eps_result', y='compound', data=df_sentiment_analysis)
plt.title('Sentiment Score by EPS Result')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'boxplot_sentiment_by_eps_result.png'))
plt.show()

# Boxplot: price change theo eps_result
plt.figure(figsize=(8, 6))
sns.boxplot(x='eps_result', y='price_change_pct', data=df_sentiment_analysis)
plt.title('Price Change % by EPS Result')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'boxplot_price_change_by_eps_result.png'))
plt.show()

# Thống kê mô tả và tương quan
print("\n[Correlation between compound and price_change_pct]")
print(df_sentiment_analysis[['compound', 'price_change_pct']].corr())

print("\n[Describe compound and price_change_pct]")
print(df_sentiment_analysis[['compound', 'price_change_pct']].describe())
