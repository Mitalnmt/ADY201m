import pandas as pd
import os

# === Config ===
TICKERS = ["AAPL"]  # Thay đổi theo nhu cầu
DATA_FOLDER = "data"

# === Function ===
def describe_file_structure(ticker):
    print(f"\n====== {ticker} ======")
    base_path = os.path.join(DATA_FOLDER, ticker)
    filenames = ["DATA_price.csv", "DATA_earnings.csv", "DATA_sentiment.csv"]

    for fname in filenames:
        fpath = os.path.join(base_path, fname)
        print(f"\n--- File: {fname} ---")

        if not os.path.exists(fpath):
            print("File not found.")
            continue

        # Nếu là price, skip 2 dòng đầu
        skip = 2 if "price" in fname else 0
        try:
            df = pd.read_csv(fpath, skiprows=skip)
            print(f"Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
            print("\nCấu trúc dữ liệu:")
            print(df.dtypes)

            print("\nThiếu dữ liệu mỗi cột:")
            print(df.isnull().sum())

            print("\nMô tả thống kê:")
            print(df.describe(include='all'))
        except Exception as e:
            print(f"Lỗi đọc file: {e}")

# === Main ===
if __name__ == "__main__":
    for tk in TICKERS:
        describe_file_structure(tk)
