import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Set style cho biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== TRẢ LỜI 10 CÂU HỎI PHÂN TÍCH DỮ LIỆU ===\n")

# ============================================================================
# 1. NHÓM ĐÃ SỬ DỤNG NHỮNG THƯ VIỆN PYTHON NÀO ĐỂ PHÂN TÍCH DỮ LIỆU?
# ============================================================================
print("1. NHÓM ĐÃ SỬ DỤNG NHỮNG THƯ VIỆN PYTHON NÀO ĐỂ PHÂN TÍCH DỮ LIỆU?")
print("=" * 80)
print("""
THƯ VIỆN CHÍNH ĐƯỢC SỬ DỤNG:

1. pandas: Xử lý và phân tích dữ liệu dạng bảng
   - Đọc/ghi CSV files
   - Thao tác với DataFrame
   - Tính toán thống kê mô tả

2. numpy: Tính toán số học và thống kê
   - Xử lý mảng dữ liệu
   - Tính toán các chỉ số thống kê
   - Xử lý dữ liệu số

3. matplotlib: Tạo biểu đồ và visualization cơ bản
   - Line plots, histograms, scatter plots
   - Customize biểu đồ
   - Export hình ảnh

4. seaborn: Tạo biểu đồ thống kê nâng cao
   - Heatmap, boxplot, histogram
   - Correlation matrix visualization
   - Statistical plotting

5. yfinance: Tải dữ liệu giá cổ phiếu từ Yahoo Finance
   - Download historical price data
   - Download earnings data
   - Real-time data access

6. sqlite3: Kết nối và truy vấn cơ sở dữ liệu SQLite
   - Database connection
   - SQL queries
   - Data storage

7. statsmodels: Phân tích chuỗi thời gian
   - seasonal_decompose: Phân rã chuỗi thời gian
   - Time series analysis

8. scikit-learn: Machine learning
   - Linear Regression
   - Random Forest
   - Model evaluation

9. scipy.stats: Thống kê nâng cao
   - Shapiro-Wilk test
   - Statistical tests

10. warnings: Xử lý cảnh báo
    - Suppress warnings
    - Error handling
""")

# ============================================================================
# 2. THỐNG KÊ MÔ TẢ (MEAN, STD, SKEWNESS, KURTOSIS) CỦA GIÁ CỔ PHIẾU
# ============================================================================
print("\n2. THỐNG KÊ MÔ TẢ (MEAN, STD, SKEWNESS, KURTOSIS) CỦA GIÁ CỔ PHIẾU")
print("=" * 80)

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}

# Tạo figure cho thống kê mô tả
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('THỐNG KÊ MÔ TẢ GIÁ CỔ PHIẾU', fontsize=16, fontweight='bold')

for i, symbol in enumerate(symbols):
    print(f"\n--- {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    
    # Lấy dữ liệu giá
    price_df = pd.read_sql_query("SELECT * FROM price", conn)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # Tính thống kê mô tả
    stats_desc = price_df['close'].describe()
    skewness = stats.skew(price_df['close'])
    kurtosis = stats.kurtosis(price_df['close'])
    
    print("Thống kê mô tả cơ bản:")
    print(stats_desc)
    print(f"\nSkewness (Độ lệch): {skewness:.4f}")
    print(f"Kurtosis (Độ nhọn): {kurtosis:.4f}")
    
    # Giải thích ý nghĩa
    if skewness > 0:
        skew_meaning = "Lệch phải (right-skewed)"
    elif skewness < 0:
        skew_meaning = "Lệch trái (left-skewed)"
    else:
        skew_meaning = "Đối xứng"
    
    if kurtosis > 3:
        kurt_meaning = "Nhọn hơn phân phối chuẩn (leptokurtic)"
    elif kurtosis < 3:
        kurt_meaning = "Phẳng hơn phân phối chuẩn (platykurtic)"
    else:
        kurt_meaning = "Tương tự phân phối chuẩn"
    
    print(f"\nÝ nghĩa:")
    print(f"- Skewness: {skew_meaning}")
    print(f"- Kurtosis: {kurt_meaning}")
    
    # Vẽ biểu đồ
    # Histogram
    axes[0, i].hist(price_df['close'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, i].axvline(price_df['close'].mean(), color='red', linestyle='--', label=f'Mean: {price_df["close"].mean():.2f}')
    axes[0, i].axvline(price_df['close'].median(), color='green', linestyle='--', label=f'Median: {price_df["close"].median():.2f}')
    axes[0, i].set_title(f'{symbol} - Phân phối giá')
    axes[0, i].set_xlabel('Giá đóng cửa')
    axes[0, i].set_ylabel('Tần suất')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, i].boxplot(price_df['close'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[1, i].set_title(f'{symbol} - Box Plot')
    axes[1, i].set_ylabel('Giá đóng cửa')
    axes[1, i].grid(True, alpha=0.3)
    
    # Thêm text thống kê
    stats_text = f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}\nStd: {price_df["close"].std():.2f}'
    axes[1, i].text(0.02, 0.98, stats_text, transform=axes[1, i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    conn.close()

plt.tight_layout()
plt.show()

# ============================================================================
# 3. TƯƠNG QUAN GIỮA GIÁ VÀ DỮ LIỆU EARNINGS
# ============================================================================
print("\n3. TƯƠNG QUAN GIỮA GIÁ VÀ DỮ LIỆU EARNINGS")
print("=" * 80)

# Tạo figure cho tương quan
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('TƯƠNG QUAN GIỮA EPS ESTIMATE VÀ REPORTED EPS', fontsize=16, fontweight='bold')

for i, symbol in enumerate(symbols):
    print(f"\n--- {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    
    # Lấy dữ liệu earnings
    earnings_df = pd.read_sql_query("SELECT * FROM earnings", conn)
    
    # Tính tương quan giữa EPS Estimate và Reported EPS
    correlation = earnings_df['EPS Estimate'].corr(earnings_df['Reported EPS'])
    
    print(f"Tương quan giữa EPS Estimate và Reported EPS: {correlation:.4f}")
    
    if correlation > 0.7:
        meaning = "Tương quan mạnh dương - Dự báo EPS khá chính xác"
    elif correlation > 0.3:
        meaning = "Tương quan trung bình dương - Dự báo EPS có độ chính xác vừa phải"
    elif correlation > -0.3:
        meaning = "Tương quan yếu - Dự báo EPS không chính xác"
    else:
        meaning = "Tương quan âm - Dự báo EPS có xu hướng ngược"
    
    print(f"Ý nghĩa: {meaning}")
    
    # Vẽ scatter plot
    axes[i].scatter(earnings_df['EPS Estimate'], earnings_df['Reported EPS'], 
                   alpha=0.7, s=100, color='coral')
    
    # Vẽ đường y=x (perfect correlation)
    min_val = min(earnings_df['EPS Estimate'].min(), earnings_df['Reported EPS'].min())
    max_val = max(earnings_df['EPS Estimate'].max(), earnings_df['Reported EPS'].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect correlation')
    
    axes[i].set_xlabel('EPS Estimate')
    axes[i].set_ylabel('Reported EPS')
    axes[i].set_title(f'{symbol}\nCorrelation: {correlation:.3f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    conn.close()

plt.tight_layout()
plt.show()

# ============================================================================
# 4. PHÂN RÃ CHUỖI THỜI GIAN CỦA GIÁ CỔ PHIẾU
# ============================================================================
print("\n4. PHÂN RÃ CHUỖI THỜI GIAN CỦA GIÁ CỔ PHIẾU")
print("=" * 80)
print("""
PHƯƠNG PHÁP PHÂN TÍCH:
- Sử dụng seasonal_decompose() từ statsmodels
- Model: 'additive' (cộng tính)
- Period: 4 (quý) cho dữ liệu theo quý

CÁC THÀNH PHẦN ĐƯỢC PHÂN TÍCH:
1. Trend (Xu hướng): Thành phần dài hạn, thể hiện hướng chung của giá
2. Seasonal (Mùa vụ): Thành phần lặp lại theo chu kỳ (4 quý = 1 năm)
3. Residual (Phần dư): Thành phần ngẫu nhiên không thể giải thích

Ý NGHĨA:
- Trend tăng: Xu hướng giá tăng dài hạn
- Trend giảm: Xu hướng giá giảm dài hạn
- Seasonal mạnh: Tính mùa vụ rõ rệt
- Residual ổn định: Dữ liệu ít nhiễu
""")

# Vẽ seasonal decomposition cho AAPL
print("\n--- Seasonal Decomposition cho AAPL ---")
conn = sqlite3.connect("aapl_analysis.db")
price_df = pd.read_sql_query("SELECT * FROM price", conn)
price_df['Date'] = pd.to_datetime(price_df['Date'])
price_df.set_index('Date', inplace=True)

# Resample theo quý
quarterly_data = price_df['close'].resample('Q').mean()

if len(quarterly_data) >= 8:
    decomposition = seasonal_decompose(quarterly_data, model='additive', period=4)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('PHÂN RÃ CHUỖI THỜI GIAN - AAPL (Quarterly)', fontsize=16, fontweight='bold')
    
    decomposition.observed.plot(ax=axes[0], title='Observed (Original Data)')
    axes[0].set_ylabel('Giá đóng cửa')
    axes[0].grid(True, alpha=0.3)
    
    decomposition.trend.plot(ax=axes[1], title='Trend (Xu hướng)')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal (Mùa vụ)')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    decomposition.resid.plot(ax=axes[3], title='Residual (Phần dư)')
    axes[3].set_ylabel('Residual')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("Không đủ dữ liệu để phân tích seasonal decomposition")

conn.close()

# ============================================================================
# 5. KẾT QUẢ PHÂN TÍCH CHÊNH LỆCH GIÁ TRƯỚC/SAU EARNINGS
# ============================================================================
print("\n5. KẾT QUẢ PHÂN TÍCH CHÊNH LỆCH GIÁ TRƯỚC/SAU EARNINGS")
print("=" * 80)

# Tạo figure cho chênh lệch giá
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('CHÊNH LỆCH GIÁ TRƯỚC/SAU EARNINGS', fontsize=16, fontweight='bold')

for i, symbol in enumerate(symbols):
    print(f"\n--- {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    
    # Truy vấn chênh lệch giá
    query = '''
    WITH price_context AS (
        SELECT e."Earnings Date" AS earnings_date,
               (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
               (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT *,
        ROUND((close_after - close_before) * 100.0 / close_before, 2) AS price_change_pct
    FROM price_context
    WHERE close_before IS NOT NULL AND close_after IS NOT NULL
    ORDER BY ABS(price_change_pct) DESC
    '''
    
    result_df = pd.read_sql_query(query, conn)
    
    print("Top 3 phản ứng giá mạnh nhất:")
    print(result_df.head(3))
    
    # Thống kê tổng quan
    mean_change = result_df['price_change_pct'].mean()
    std_change = result_df['price_change_pct'].std()
    max_change = result_df['price_change_pct'].max()
    min_change = result_df['price_change_pct'].min()
    
    print(f"\nThống kê tổng quan:")
    print(f"- Thay đổi trung bình: {mean_change:.2f}%")
    print(f"- Độ lệch chuẩn: {std_change:.2f}%")
    print(f"- Thay đổi lớn nhất: {max_change:.2f}%")
    print(f"- Thay đổi nhỏ nhất: {min_change:.2f}%")
    
    # Vẽ biểu đồ
    # Bar chart cho top 5 phản ứng
    top_5 = result_df.head(5)
    colors = ['red' if x < 0 else 'green' for x in top_5['price_change_pct']]
    
    bars = axes[i].bar(range(len(top_5)), top_5['price_change_pct'], color=colors, alpha=0.7)
    axes[i].set_xlabel('Thứ tự sự kiện')
    axes[i].set_ylabel('Thay đổi giá (%)')
    axes[i].set_title(f'{symbol} - Top 5 Phản ứng giá')
    axes[i].grid(True, alpha=0.3)
    
    # Thêm giá trị trên bars
    for bar, value in zip(bars, top_5['price_change_pct']):
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Thêm thống kê
    stats_text = f'Mean: {mean_change:.2f}%\nStd: {std_change:.2f}%\nMax: {max_change:.2f}%\nMin: {min_change:.2f}%'
    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    conn.close()

plt.tight_layout()
plt.show()

# ============================================================================
# 6. PHÂN TÍCH ĐỂ XÁC ĐỊNH TÁC ĐỘNG CỦA EARNINGS
# ============================================================================
print("\n6. PHÂN TÍCH ĐỂ XÁC ĐỊNH TÁC ĐỘNG CỦA EARNINGS")
print("=" * 80)
print("""
CÁC PHÂN TÍCH ĐƯỢC THỰC HIỆN:

1. PHÂN TÍCH TƯƠNG QUAN:
   - EPS Estimate vs Reported EPS
   - Sentiment vs Price Change
   - Correlation matrix analysis

2. PHÂN TÍCH PHẢN ỨNG GIÁ:
   - 1 ngày trước/sau earnings
   - 3 ngày trước/sau earnings
   - Tính toán phần trăm thay đổi

3. PHÂN TÍCH SENTIMENT:
   - Compound sentiment score
   - Tương quan với thay đổi giá
   - Time series analysis

4. PHÂN TÍCH CLUSTERING:
   - Nhóm sự kiện theo mức độ phản ứng
   - Strong (≥5%), Medium (2-5%), Weak (<2%)

5. MACHINE LEARNING:
   - Linear Regression
   - Random Forest
   - Dự báo thay đổi giá dựa trên sentiment

6. WINDOW FUNCTIONS:
   - Moving averages
   - Rolling statistics
   - Trend analysis
""")

# ============================================================================
# 7. KIỂM TRA PHÂN PHỐI CHUẨN CỦA DỮ LIỆU GIÁ
# ============================================================================
print("\n7. KIỂM TRA PHÂN PHỐI CHUẨN CỦA DỮ LIỆU GIÁ")
print("=" * 80)

# Tạo figure cho kiểm tra phân phối chuẩn
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('KIỂM TRA PHÂN PHỐI CHUẨN', fontsize=16, fontweight='bold')

for i, symbol in enumerate(symbols):
    print(f"\n--- {symbol} ---")
    conn = sqlite3.connect(db_names[symbol])
    
    # Lấy dữ liệu giá
    price_df = pd.read_sql_query("SELECT * FROM price", conn)
    
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(price_df['close'])
    
    print(f"Shapiro-Wilk Test:")
    print(f"- Statistic: {statistic:.4f}")
    print(f"- P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        result = "KHÔNG từ chối H0 - Dữ liệu có thể tuân theo phân phối chuẩn"
    else:
        result = "Từ chối H0 - Dữ liệu KHÔNG tuân theo phân phối chuẩn"
    
    print(f"Kết luận: {result}")
    
    # Tính skewness và kurtosis
    skewness = stats.skew(price_df['close'])
    kurtosis = stats.kurtosis(price_df['close'])
    
    print(f"\nSkewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    
    print("\nẢNH HƯỞNG:")
    if abs(skewness) > 1 or abs(kurtosis) > 3:
        print("- Dữ liệu không chuẩn → Cần sử dụng phương pháp phi tham số")
        print("- Có thể cần transform dữ liệu (log, square root)")
        print("- Ảnh hưởng đến việc chọn thuật toán ML")
    else:
        print("- Dữ liệu gần chuẩn → Có thể sử dụng phương pháp tham số")
    
    # Vẽ biểu đồ
    # Q-Q Plot
    stats.probplot(price_df['close'], dist="norm", plot=axes[0, i])
    axes[0, i].set_title(f'{symbol} - Q-Q Plot')
    axes[0, i].grid(True, alpha=0.3)
    
    # Histogram với đường chuẩn
    axes[1, i].hist(price_df['close'], bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Vẽ đường phân phối chuẩn
    x = np.linspace(price_df['close'].min(), price_df['close'].max(), 100)
    normal_dist = stats.norm.pdf(x, price_df['close'].mean(), price_df['close'].std())
    axes[1, i].plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
    
    axes[1, i].set_title(f'{symbol} - Histogram vs Normal')
    axes[1, i].set_xlabel('Giá đóng cửa')
    axes[1, i].set_ylabel('Density')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)
    
    # Thêm thông tin test
    test_text = f'Shapiro-Wilk:\np-value: {p_value:.4f}\n{"Normal" if p_value > 0.05 else "Not Normal"}'
    axes[1, i].text(0.02, 0.98, test_text, transform=axes[1, i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    conn.close()

plt.tight_layout()
plt.show()

# ============================================================================
# 8. PHÁT HIỆN HỮU ÍCH CHO DỰ BÁO GIÁ
# ============================================================================
print("\n8. PHÁT HIỆN HỮU ÍCH CHO DỰ BÁO GIÁ")
print("=" * 80)
print("""
CÁC PHÁT HIỆN QUAN TRỌNG:

1. MỐI TƯƠNG QUAN GIỮA SENTIMENT VÀ THAY ĐỔI GIÁ:
   - Sentiment cao → Thường có phản ứng giá tích cực
   - Sentiment thấp → Thường có phản ứng giá tiêu cực
   - Có thể sử dụng sentiment làm feature cho ML model

2. PATTERN THEO MÙA VỤ:
   - Phản ứng giá có tính chu kỳ theo quý
   - Một số quý có phản ứng mạnh hơn quý khác
   - Có thể dựa vào pattern này để dự báo

3. TÁC ĐỘNG KHÁC NHAU CỦA EARNINGS EVENTS:
   - Beat earnings → Thường có phản ứng giá tích cực
   - Miss earnings → Thường có phản ứng giá tiêu cực
   - Meet earnings → Phản ứng ít rõ ràng hơn

4. ĐỘ CHÍNH XÁC CỦA DỰ BÁO EPS:
   - Tương quan cao giữa estimate và actual → Dự báo tốt
   - Tương quan thấp → Thị trường bất ngờ nhiều

5. PHÂN LOẠI MỨC ĐỘ TÁC ĐỘNG:
   - Strong reaction events → Cần chú ý đặc biệt
   - Medium reaction events → Tác động vừa phải
   - Weak reaction events → Ít tác động

ỨNG DỤNG CHO DỰ BÁO:
- Sử dụng sentiment làm leading indicator
- Kết hợp với technical analysis
- Phân tích pattern theo mùa vụ
- Machine learning với multiple features
""")

# ============================================================================
# 9. LƯU TRỮ KẾT QUẢ PHÂN TÍCH
# ============================================================================
print("\n9. LƯU TRỮ KẾT QUẢ PHÂN TÍCH")
print("=" * 80)
print("""
CÁC VỊ TRÍ LƯU TRỮ:

1. CSV FILES:
   - aapl_price.csv, msft_price.csv, googl_price.csv
   - aapl_earnings.csv, msft_earnings.csv, googl_earnings.csv
   - aapl_sentiment.csv, msft_sentiment.csv, googl_sentiment.csv
   - Dữ liệu thô và đã xử lý

2. SQLITE DATABASES:
   - aapl_analysis.db
   - msft_analysis.db
   - googl_analysis.db
   - Mỗi database chứa 3 bảng: price, earnings, sentiment

3. CẤU TRÚC DATABASE:
   - Bảng price: Date, open, high, low, close, volume
   - Bảng earnings: Earnings Date, EPS Estimate, Reported EPS
   - Bảng sentiment: Earnings Date, compound, positive, negative, neutral

4. PYTHON SCRIPTS:
   - Các file phân tích và visualization
   - Code để reproduce kết quả
   - Documentation và comments

5. VISUALIZATION OUTPUTS:
   - Biểu đồ được lưu dưới dạng PNG/JPG
   - Heatmaps, histograms, time series plots
   - Statistical charts
""")

# ============================================================================
# 10. THÁCH THỨC KHI PHÂN TÍCH DỮ LIỆU BẰNG PYTHON
# ============================================================================
print("\n10. THÁCH THỨC KHI PHÂN TÍCH DỮ LIỆU BẰNG PYTHON")
print("=" * 80)
print("""
CÁC THÁCH THỨC CHÍNH:

1. MISSING DATA:
   - Dữ liệu thiếu trong price và earnings
   - Cần xử lý bằng forward fill, backward fill, hoặc interpolation
   - Ảnh hưởng đến tính chính xác của phân tích

2. DATE ALIGNMENT:
   - Đồng bộ ngày tháng giữa các bảng
   - Xử lý timezone differences
   - Đảm bảo tính nhất quán của dữ liệu

3. OUTLIER DETECTION:
   - Xác định giá trị bất thường
   - Quyết định có loại bỏ hay giữ lại
   - Ảnh hưởng đến kết quả thống kê

4. DATA QUALITY:
   - Đảm bảo tính nhất quán của dữ liệu
   - Kiểm tra format và type của dữ liệu
   - Validation dữ liệu đầu vào

5. PERFORMANCE:
   - Tối ưu hóa truy vấn SQL cho dataset lớn
   - Memory management cho large datasets
   - Processing time optimization

6. VISUALIZATION:
   - Tạo biểu đồ phù hợp cho từng loại phân tích
   - Customize appearance và style
   - Export high-quality images

7. MODEL SELECTION:
   - Chọn thuật toán ML phù hợp với đặc điểm dữ liệu
   - Hyperparameter tuning
   - Model evaluation và validation

8. ERROR HANDLING:
   - Xử lý exceptions và errors
   - Logging và debugging
   - Robust code development

9. REPRODUCIBILITY:
   - Đảm bảo kết quả có thể reproduce
   - Version control cho code và data
   - Documentation đầy đủ

10. INTERPRETATION:
    - Giải thích kết quả thống kê
    - Business insights từ technical analysis
    - Communication với stakeholders
""")

print("\n" + "="*80)
print("HOÀN THÀNH TRẢ LỜI 10 CÂU HỎI PHÂN TÍCH DỮ LIỆU")
print("="*80) 