import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# ========== Config ==========
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

DATA_FOLDER = "data"

# ========== Hàm xử lý ==========
def advanced_preprocessing(ticker):
    print(f"\n--- Xử lý dữ liệu cho: {ticker} ---")
    fpath = os.path.join(DATA_FOLDER, ticker, "DATA_price.csv")
    if not os.path.exists(fpath):
        print("Không tìm thấy file price.")
        return

    df = pd.read_csv(fpath, skiprows=2)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)

    # 1. Missing value: interpolation + KNN imputation
    df.interpolate(method='linear', inplace=True)
    imputer = KNNImputer(n_neighbors=3)
    df[df.columns] = imputer.fit_transform(df)

    # 2. Outlier removal: Z-score + IQR (2 bước)
    z_thresh = 3
    z_scores = np.abs((df - df.mean()) / df.std())
    df = df[(z_scores < z_thresh).all(axis=1)]

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # 3. Feature engineering
    df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
    df['rolling_std_5'] = df['Close'].rolling(window=5).std()
    df['volatility'] = df['High'] - df['Low']
    df['price_change_pct'] = df['Close'].pct_change() * 100

    df.dropna(inplace=True)

    # 4. Normalize
    scaler = MinMaxScaler()
    scaled_df = df.copy()
    scaled_df[df.columns] = scaler.fit_transform(df[df.columns])

    # Save lại
    out_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}_processed.csv")
    scaled_df.to_csv(out_path)
    print(f"✅ Đã lưu: {out_path}")

# ========== Main ==========
if __name__ == "__main__":
    for tk in TICKERS:
        advanced_preprocessing(tk)
