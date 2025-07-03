import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# === Tạo thư mục lưu hình nếu chưa có ===
chart_dir = os.path.join("chart", "4 sentiment")
os.makedirs(chart_dir, exist_ok=True)

# === Kết nối và truy vấn dữ liệu ===
conn = sqlite3.connect('aapl_analysis.db')

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
sentiment_df = pd.read_sql_query("SELECT * FROM sentiment", conn)
conn.close()

# Gộp dữ liệu
df_sentiment_analysis = pd.merge(df_analysis, sentiment_df, left_on='earnings_date', right_on='Earnings Date')
df_sentiment_analysis['earnings_date'] = pd.to_datetime(df_sentiment_analysis['earnings_date'])
df_sentiment_analysis.set_index('earnings_date', inplace=True)

# === 1. Mô tả cơ bản ===
print("Mô tả cơ bản:")
print(df_sentiment_analysis[['compound', 'price_change_pct']].describe())

# === 2. Ma trận tương quan ===
print("\nMa trận tương quan:")
corr_matrix = df_sentiment_analysis[['compound', 'price_change_pct']].corr()
print(corr_matrix)

plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, "correlation_matrix.png"))
plt.show()

# === 3. Trung bình theo quý ===
quarterly_avg = df_sentiment_analysis[['compound', 'price_change_pct']].resample('Q').mean()

print("\nTrung bình hàng quý:")
print(quarterly_avg)

plt.figure(figsize=(12, 5))
plt.plot(quarterly_avg.index, quarterly_avg['compound'], marker='o', label='Avg Sentiment (compound)')
plt.plot(quarterly_avg.index, quarterly_avg['price_change_pct'], marker='o', label='Avg Price Change (%)')
plt.title("Quarterly Average Sentiment and Price Change")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, "quarterly_avg_sentiment_price_change.png"))
plt.show()

# === 4. Phân tích chu kỳ bằng seasonal_decompose ===
series = quarterly_avg['compound'].dropna()

result = seasonal_decompose(series, model='additive', period=4)
plt.figure()
result.plot()
plt.suptitle("Sentiment Time Series Decomposition (Quarterly)")
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, "sentiment_seasonal_decompose.png"))
plt.show()
