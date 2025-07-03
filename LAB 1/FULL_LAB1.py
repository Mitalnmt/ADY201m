import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ================== Config ==================
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
START_DATE = "2023-01-01"
END_DATE = "2025-05-31"
DB_PATH = "data/multi_stock_analysis.db"
sns.set(style="whitegrid")

# ================== 1. Get Data ==================
def fetch_all_data(ticker: str) -> None:
    folder = f"data/{ticker}"
    os.makedirs(folder, exist_ok=True)

    price = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)
    price.to_csv(f"{folder}/DATA_price.csv")

    t = yf.Ticker(ticker)
    earnings = t.earnings_dates.reset_index()
    earnings["Earnings Date"] = pd.to_datetime(earnings["Earnings Date"])
    earnings = earnings.set_index("Earnings Date").sort_index()[START_DATE:END_DATE]
    earnings.to_csv(f"{folder}/DATA_earnings.csv")

    sentiment = earnings.reset_index()
    sentiment["compound"] = np.random.uniform(-1, 1, size=len(sentiment))
    sentiment[["Earnings Date", "compound"]].to_csv(f"{folder}/DATA_sentiment.csv", index=False)

    _fix_header_third_line(f"{folder}/DATA_price.csv")

def _fix_header_third_line(csv_path: str) -> None:
    with open(csv_path, "r") as f:
        lines = f.readlines()
    if len(lines) >= 3:
        lines[2] = "Date,Close,High,Low,Open,Volume\n"
        with open(csv_path, "w") as f:
            f.writelines(lines)

# ================== 2. Preprocess ==================
def preprocess_and_scale(ticker: str) -> None:
    path = f"data/{ticker}/DATA_price.csv"
    df = pd.read_csv(path, skiprows=2)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df.set_index("Date", inplace=True)
    df.ffill(inplace=True)

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[df.columns] = scaler.fit_transform(df)
    df_scaled.to_csv(f"data/{ticker}/{ticker}_scaled.csv")

# ================== 3. Import to DB ==================
def import_to_db(ticker: str) -> None:
    folder = f"data/{ticker}"
    conn = sqlite3.connect(DB_PATH)

    price = pd.read_csv(f"{folder}/DATA_price.csv", skiprows=2)
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price.dropna(subset=["Date"], inplace=True)
    price["Date"] = price["Date"].dt.strftime("%Y-%m-%d")

    earnings = pd.read_csv(f"{folder}/DATA_earnings.csv")
    earnings["Earnings Date"] = pd.to_datetime(earnings["Earnings Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.strftime("%Y-%m-%d")

    sentiment = pd.read_csv(f"{folder}/DATA_sentiment.csv")
    sentiment["Earnings Date"] = pd.to_datetime(sentiment["Earnings Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.strftime("%Y-%m-%d")

    price.to_sql(f"{ticker}_price", conn, if_exists="replace", index=False)
    earnings.to_sql(f"{ticker}_earnings", conn, if_exists="replace", index=False)
    sentiment.to_sql(f"{ticker}_sentiment", conn, if_exists="replace", index=False)
    conn.close()

# ================== 4. Analyze ==================
def analyze_and_model(ticker: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    WITH price_context AS (
        SELECT 
            e."Earnings Date" AS earnings_date,
            e."EPS Estimate" AS eps_est,
            e."Reported EPS" AS eps_actual,
            (SELECT p1.Close FROM {ticker}_price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
            (SELECT p2.Close FROM {ticker}_price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM {ticker}_earnings e
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
    df_eps = pd.read_sql_query(query, conn)
    sentiment = pd.read_sql_query(f"SELECT * FROM {ticker}_sentiment", conn)
    conn.close()

    df = pd.merge(df_eps, sentiment, left_on="earnings_date", right_on="Earnings Date")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df.set_index("earnings_date", inplace=True)

    if len(df) < 10:
        print(f"{ticker} – only {len(df)} merged rows (<10). Skipping ML step to avoid empty train/test split.")
        return

    chart_dir = f"chart/{ticker}"
    os.makedirs(chart_dir, exist_ok=True)

    plt.figure()
    sns.scatterplot(data=df, x="compound", y="price_change_pct")
    plt.title(f"{ticker} – Sentiment vs Price Change")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/scatter_sentiment_vs_price.png")
    plt.close()

    plt.figure()
    sns.boxplot(x="eps_result", y="compound", data=df)
    plt.title(f"{ticker} – Sentiment by EPS Result")
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/box_sentiment_by_eps.png")
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        df[["compound"]], df["price_change_pct"], test_size=0.2, random_state=42
    )

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

    rmse_lr = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))

    print(f"{ticker} – RMSE Linear: {rmse_lr:.2f} | Random Forest: {rmse_rf:.2f}")

    df.reset_index().to_csv(f"data/{ticker}/Data_sentiment_analysis.csv", index=False)

# ================== Main ==================
if __name__ == "__main__":
    for tk in TICKERS:
        print(f"\n=== Processing {tk} ===")
        fetch_all_data(tk)
        preprocess_and_scale(tk)
        import_to_db(tk)
        analyze_and_model(tk)
