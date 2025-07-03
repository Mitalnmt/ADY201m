import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.patches as mpatches
import os

# === Tạo thư mục lưu hình ===
chart_dir = os.path.join("chart", "5 Data Visualization")
os.makedirs(chart_dir, exist_ok=True)

# === Kết nối DB và đọc dữ liệu ===
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

df_sentiment_analysis = pd.merge(df_analysis, sentiment_df, left_on='earnings_date', right_on='Earnings Date')
df_sentiment_analysis['earnings_date'] = pd.to_datetime(df_sentiment_analysis['earnings_date'])
df_sentiment_analysis.set_index('earnings_date', inplace=True)

# === 1. Histogram sentiment ===
plt.figure()
sns.histplot(df_sentiment_analysis['compound'], bins=10, kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Compound Sentiment')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'hist_sentiment_distribution.png'))
plt.show()

# === 2. Line plot theo tháng ===
monthly_avg = df_sentiment_analysis[['compound', 'price_change_pct']].resample('ME').mean()
monthly_clean = monthly_avg.dropna()

plt.figure(figsize=(12,5))
plt.plot(monthly_clean.index, monthly_clean['compound'], label='Avg Sentiment (compound)', marker='o')
plt.plot(monthly_clean.index, monthly_clean['price_change_pct'], label='Avg Price Change (%)', marker='s')
plt.title("Monthly Average Sentiment and Price Change")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'monthly_avg_sentiment_price.png'))
plt.show()

# === 3. Line plot theo earnings_date ===
df_plot = df_sentiment_analysis.reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(data=df_plot, x='earnings_date', y='price_change_pct', marker='o', label='Price Change (%)')
sns.lineplot(data=df_plot, x='earnings_date', y='compound', marker='s', label='Sentiment (compound)')
plt.title("Sentiment and Price Change per Earnings Report")
plt.xlabel("Earnings Date")
plt.ylabel("Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'line_sentiment_price_by_earnings.png'))
plt.show()

# === 4. Bar Chart sentiment theo eps_result ===
plt.figure()
sns.barplot(data=df_sentiment_analysis, x='eps_result', y='compound', estimator='mean', hue='eps_result', palette='Set3', legend=False)
plt.title('Avg Sentiment by EPS Result')
plt.xlabel('EPS Result')
plt.ylabel('Avg Compound Sentiment')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'bar_sentiment_by_eps.png'))
plt.show()

# === 5. Scatter plot sentiment vs price change ===
plt.figure()
sns.scatterplot(data=df_sentiment_analysis, x='compound', y='price_change_pct', hue='eps_result')
plt.title('Sentiment vs Price Change')
plt.xlabel('Compound Sentiment')
plt.ylabel('Price Change (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'scatter_sentiment_vs_price.png'))
plt.show()

# === 6. Heatmap correlation ===
print(df_sentiment_analysis.columns)
print(df_sentiment_analysis[['compound', 'price_change_pct']].head())
corr_matrix = df_sentiment_analysis[['compound', 'price_change_pct']].corr()
plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'heatmap_correlation.png'))
plt.show()

# === 7. Seasonal Decomposition ===
if monthly_avg['compound'].dropna().shape[0] >= 8:
    result = seasonal_decompose(monthly_avg['compound'].dropna(), model='additive', period=4)
    plt.figure()
    result.plot()
    plt.suptitle('Seasonal Decomposition of Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'seasonal_decompose_sentiment.png'))
    plt.show()

# === 8. Boxplot sentiment theo eps_result ===
plt.figure()
sns.boxplot(data=df_sentiment_analysis, x='eps_result', y='compound', hue='eps_result', palette='Set2', legend=False)
plt.title('Sentiment Distribution by EPS Result')
plt.xlabel('EPS Result')
plt.ylabel('Compound Sentiment')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'boxplot_sentiment_eps.png'))
plt.show()

# === 9. Line plot giá trước/sau ===
plt.figure()
df_sentiment_analysis_sorted = df_sentiment_analysis.sort_index()
plt.plot(df_sentiment_analysis_sorted.index, df_sentiment_analysis_sorted['close_before'], label='Close Before')
plt.plot(df_sentiment_analysis_sorted.index, df_sentiment_analysis_sorted['close_after'], label='Close After')
plt.title('Stock Price Before and After Earnings')
plt.xlabel('Earnings Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'line_price_before_after.png'))
plt.show()

# === 10. Waffle-style bar chart EPS Result Distribution ===
counts = df_sentiment_analysis['eps_result'].value_counts()
labels = counts.index.tolist()
values = counts.values
colors = ['#4CAF50', '#FFC107', '#F44336']
fig, ax = plt.subplots()
start = 0
for i in range(len(values)):
    ax.barh(0, values[i], left=start, color=colors[i])
    start += values[i]
ax.set_xlim(0, sum(values))
legend = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend, loc='upper center')
plt.title('EPS Result Distribution (Waffle Style)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(chart_dir, 'waffle_eps_result.png'))
plt.show()
