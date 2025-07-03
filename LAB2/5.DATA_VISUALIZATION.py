import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.patches as mpatches

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

for symbol in symbols:
    print(f"\n--- Data Visualization for {symbol} ---")
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
    df_sentiment_analysis = pd.merge(df_analysis, sentiment_df, left_on='earnings_date', right_on='Earnings Date')
    df_sentiment_analysis['earnings_date'] = pd.to_datetime(df_sentiment_analysis['earnings_date'])
    df_sentiment_analysis.set_index('earnings_date', inplace=True)
    # 1. Histogram sentiment
    plt.figure()
    sns.histplot(df_sentiment_analysis['compound'], bins=10, kde=True)
    plt.title(f'Distribution of Sentiment Scores ({symbol})')
    plt.xlabel('Compound Sentiment')
    plt.ylabel('Frequency')
    plt.show()
    # 2. Line plot sentiment theo thời gian
    monthly_avg = df_sentiment_analysis[['compound', 'price_change_pct']].resample('ME').mean()
    monthly_clean = monthly_avg.dropna()
    plt.figure(figsize=(12,5))
    plt.plot(monthly_clean.index, monthly_clean['compound'], label='Avg Sentiment (compound)', marker='o')
    plt.plot(monthly_clean.index, monthly_clean['price_change_pct'], label='Avg Price Change (%)', marker='s')
    plt.title(f"Monthly Average Sentiment and Price Change ({symbol})")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 3. Line plot theo earnings_date
    df_plot = df_sentiment_analysis.reset_index()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_plot, x='earnings_date', y='price_change_pct', marker='o', label='Price Change (%)')
    sns.lineplot(data=df_plot, x='earnings_date', y='compound', marker='s', label='Sentiment (compound)')
    plt.title(f"Sentiment and Price Change per Earnings Report ({symbol})")
    plt.xlabel("Earnings Date")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # 4. Bar Chart sentiment theo eps_result
    plt.figure()
    sns.barplot(data=df_sentiment_analysis, x='eps_result', y='compound', estimator='mean', hue='eps_result', palette='Set3', legend=False)
    plt.title(f'Avg Sentiment by EPS Result ({symbol})')
    plt.xlabel('EPS Result')
    plt.ylabel('Avg Compound Sentiment')
    plt.show()
    # 5. Scatter plot sentiment vs price change
    plt.figure()
    sns.scatterplot(data=df_sentiment_analysis, x='compound', y='price_change_pct', hue='eps_result')
    plt.title(f'Sentiment vs Price Change ({symbol})')
    plt.xlabel('Compound Sentiment')
    plt.ylabel('Price Change (%)')
    plt.legend()
    plt.show()
    # 6. Heatmap correlation
    corr_matrix = df_sentiment_analysis[['compound', 'price_change_pct']].corr()
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix ({symbol})')
    plt.show()
    # 7. Seasonal decomposition (nếu đủ dữ liệu)
    if monthly_avg['compound'].dropna().shape[0] >= 8:
        result = seasonal_decompose(monthly_avg['compound'].dropna(), model='additive', period=4)
        result.plot()
        plt.suptitle(f'Seasonal Decomposition of Sentiment ({symbol})')
        plt.show()
    # 8. Boxplot sentiment theo eps_result
    plt.figure()
    sns.boxplot(data=df_sentiment_analysis, x='eps_result', y='compound', hue='eps_result', palette='Set2', legend=False)
    plt.title(f'Sentiment Distribution by EPS Result ({symbol})')
    plt.xlabel('EPS Result')
    plt.ylabel('Compound Sentiment')
    plt.show()
    # 9. Line plot giá trước/sau
    plt.figure()
    df_sentiment_analysis_sorted = df_sentiment_analysis.sort_index()
    plt.plot(df_sentiment_analysis_sorted.index, df_sentiment_analysis_sorted['close_before'], label='Close Before')
    plt.plot(df_sentiment_analysis_sorted.index, df_sentiment_analysis_sorted['close_after'], label='Close After')
    plt.title(f'Stock Price Before and After Earnings ({symbol})')
    plt.xlabel('Earnings Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.show()
    # 10. Waffle chart eps_result (nếu thư viện phù hợp)
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
    plt.title(f'EPS Result Distribution (Waffle Style) - {symbol}')
    plt.axis('off')
    plt.show() 