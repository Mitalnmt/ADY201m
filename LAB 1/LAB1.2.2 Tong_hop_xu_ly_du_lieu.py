# === 2.4 Tien_xu_ly_du_lieu.py ===
import pandas as pd
import sys

try:
    # Đọc dữ liệu, bỏ 2 dòng đầu không hợp lệ
    df = pd.read_csv('data/AAPL_price.csv', skiprows=2)

    # Đổi tên cột đầu tiên nếu cần
    df.rename(columns={'Price': 'Adj Close'}, inplace=True)

    # Chuyển cột Date sang dạng datetime
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
    print("Lỗi: Không tìm thấy file data/data/AAPL_price.csv")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi không xác định: {str(e)}")
    sys.exit(1)




# === 2.5 Mo_ta_cau_truc_du_lieu.py ===
import pandas as pd

# Đọc dữ liệu, bỏ 2 dòng đầu không hợp lệ
df = pd.read_csv('data/AAPL_price.csv', skiprows=2)

# Đổi tên cột đầu tiên nếu cần
df.rename(columns={'Price': 'Adj Close'}, inplace=True)

# Chuyển cột Date sang dạng datetime
df['Date'] = pd.to_datetime(df['Date'])

# Đặt cột Date làm chỉ mục
df.set_index('Date', inplace=True)

# Xem thông tin tổng quát
print(df.info())
print(df.describe())


# === 2.6 Min_Max Scaling.py ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

try:
    # Đọc dữ liệu đã được xử lý từ file trước
    df = pd.read_csv('data/AAPL_price.csv', skiprows=2)
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
    scaled_df.to_csv('data/AAPL_scaled.csv')
    print("\nĐã lưu kết quả vào file AAPL_scaled.csv")

except FileNotFoundError:
    print("Lỗi: Không tìm thấy file data/data/AAPL_price.csv")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi không xác định: {str(e)}")
    sys.exit(1)


# === 2.7 xu ly outliers.py ===
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the scaled data
df = pd.read_csv('data/AAPL_scaled.csv')

plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Open', 'Close', 'Volume']])
plt.title("Boxplot - phát hiện outliers (dữ liệu đã scale)")
plt.xlabel('Open (4), Close (1), Volume (5)')
plt.show()


