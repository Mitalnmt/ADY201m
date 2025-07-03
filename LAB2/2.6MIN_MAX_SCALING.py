import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

files = ["AAPL_price.csv", "MSFT_price.csv", "GOOGL_price.csv"]

for file in files:
    try:
        print(f"\n--- Xử lý file: {file} ---")
        # Đọc dữ liệu đã được xử lý từ file trước
        df = pd.read_csv(file, skiprows=2)
        # Đổi tên cột đầu tiên nếu cần
        df.rename(columns={'Price': 'Adj Close'}, inplace=True)
        # Chuyển cột Date sang dạng datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Đặt cột Date làm chỉ mục
        df.set_index('Date', inplace=True)
        # Khởi tạo scaler
        scaler = MinMaxScaler()
        # Tạo bản sao của dataframe
        scaled_df = df.copy()
        # Lấy tên các cột số
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        # Áp dụng Min-Max scaling cho các cột số
        scaled_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        # Hiển thị kết quả
        print("\nDữ liệu gốc:")
        print(df.head())
        print("\nDữ liệu sau khi scaling:")
        print(scaled_df.head())
        # Lưu kết quả
        out_file = file.replace('_price.csv', '_scaled.csv')
        scaled_df.to_csv(out_file)
        print(f"\nĐã lưu kết quả vào file {out_file}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file}")
        continue
    except Exception as e:
        print(f"Lỗi không xác định: {str(e)}")
        continue 