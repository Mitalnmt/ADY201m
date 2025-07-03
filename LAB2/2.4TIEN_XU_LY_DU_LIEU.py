import pandas as pd
import sys

files = ["AAPL_price.csv", "MSFT_price.csv", "GOOGL_price.csv"]

for file in files:
    print(f"\n--- Xử lý file: {file} ---")
    try:
        # Đọc dữ liệu, bỏ 2 dòng đầu không hợp lệ
        df = pd.read_csv(file, skiprows=2)

        # Đổi tên cột đầu tiên nếu cần
        if 'Price' in df.columns:
            df.rename(columns={'Price': 'Adj Close'}, inplace=True)

        # Chuyển cột Date sang dạng datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            # Đặt cột Date làm chỉ mục
            df.set_index('Date', inplace=True)

        # Kiểm tra missing values
        print("\nSố lượng missing values trong mỗi cột:")
        print(df.isnull().sum())

        # Điền missing values bằng phương pháp forward fill
        df.fillna(method='ffill', inplace=True)

        # Kiểm tra lại sau khi điền
        print("\nSố lượng missing values sau khi điền:")
        print(df.isnull().sum())

        # Hiển thị thông tin tổng quát về dữ liệu
        print("\nThông tin tổng quát về dữ liệu:")
        print(df.info())

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file}")
        continue
    except Exception as e:
        print(f"Lỗi không xác định: {str(e)}")
        continue 