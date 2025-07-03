import pandas as pd

files = ["AAPL_scaled.csv", "MSFT_scaled.csv", "GOOGL_scaled.csv"]
labels = ["AAPL", "MSFT", "GOOGL"]

for file, label in zip(files, labels):
    print(f"\n--- Feature engineering cho {label} ---")
    df = pd.read_csv(file)
    # Nếu có cột Date, chuyển sang datetime và set index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    # Rolling mean 5 ngày cho cột 'Unnamed: 1' (Close)
    df['rolling_mean_5'] = df['Unnamed: 1'].rolling(window=5).mean()
    # Volatility 5 ngày cho cột 'Unnamed: 1' (Close)
    df['volatility_5'] = df['Unnamed: 1'].rolling(window=5).std()
    # % thay đổi giá (Close)
    df['pct_change'] = df['Unnamed: 1'].pct_change()
    # Lưu ra file mới
    out_file = file.replace('_scaled.csv', '_feature.csv')
    df.to_csv(out_file)
    print(f"Đã lưu file: {out_file}") 