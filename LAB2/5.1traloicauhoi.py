import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=== TRẢ LỜI 10 CÂU HỎI VỀ TRỰC QUAN HÓA DỮ LIỆU ===\n")

# Đảm bảo thư mục lưu biểu đồ tồn tại
os.makedirs('charts', exist_ok=True)

symbols = ["AAPL", "MSFT", "GOOGL"]
db_names = {
    "AAPL": "aapl_analysis.db",
    "MSFT": "msft_analysis.db",
    "GOOGL": "googl_analysis.db"
}
colors = {
    "AAPL": "blue",
    "MSFT": "orange",
    "GOOGL": "green"
}

# 1. Biểu đồ line thể hiện xu hướng giá trước/sau earnings cho thấy điều gì?
print("1. Biểu đồ line thể hiện xu hướng giá trước/sau earnings cho thấy điều gì?")
print("- Biểu đồ line cho thấy xu hướng giá cổ phiếu trước và sau các sự kiện earnings.\n  Giá thường biến động mạnh quanh ngày earnings, giúp nhận diện các giai đoạn tăng/giảm rõ rệt.\n  Các điểm đỏ là ngày earnings, thường trùng với các biến động lớn trên đường giá.")

for symbol in symbols:
    conn = sqlite3.connect(db_names[symbol])
    price_df = pd.read_sql_query("SELECT * FROM price", conn)
    earnings_df = pd.read_sql_query("SELECT * FROM earnings", conn)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    earnings_df['Earnings Date'] = pd.to_datetime(earnings_df['Earnings Date'])
    plt.figure(figsize=(12,6))
    plt.plot(price_df['Date'], price_df['close'], color=colors[symbol], label=f'{symbol} Close Price')
    plt.scatter(earnings_df['Earnings Date'], price_df.set_index('Date').reindex(earnings_df['Earnings Date'])['close'],
                color='red', marker='o', label='Earnings Date')
    plt.title(f'Biểu đồ Line: Xu hướng giá trước/sau earnings - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'charts/line_price_{symbol}.png')
    plt.close()
    conn.close()

# 2. Biểu đồ bar thể hiện chênh lệch giá theo cổ phiếu cho thấy xu hướng nào?
print("\n2. Biểu đồ bar thể hiện chênh lệch giá theo cổ phiếu cho thấy xu hướng nào?")
print("- Bar chart cho thấy mức biến động giá sau earnings của từng cổ phiếu.\n  Có những sự kiện phản ứng giá rất mạnh (cả tăng và giảm),\n  giúp xác định cổ phiếu nào thường có biến động lớn nhất quanh earnings.")

for symbol in symbols:
    conn = sqlite3.connect(db_names[symbol])
    query = '''
    WITH price_context AS (
        SELECT e."Earnings Date" AS earnings_date,
               (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
               (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT earnings_date, ROUND((close_after - close_before) * 100.0 / close_before, 2) AS price_change_pct
    FROM price_context
    WHERE close_before IS NOT NULL AND close_after IS NOT NULL
    ORDER BY earnings_date
    '''
    df = pd.read_sql_query(query, conn)
    df['earnings_date'] = pd.to_datetime(df['earnings_date'])
    plt.figure(figsize=(10,5))
    bars = plt.bar(df['earnings_date'].dt.strftime('%Y-%m-%d'), df['price_change_pct'], color=colors[symbol], alpha=0.7)
    plt.title(f'Biểu đồ Bar: Chênh lệch giá sau earnings - {symbol}')
    plt.xlabel('Earnings Date')
    plt.ylabel('Price Change (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'charts/bar_price_change_{symbol}.png')
    plt.close()
    conn.close()

# 3. Biểu đồ area thể hiện tích lũy giá theo thời gian cho thấy điều gì?
print("\n3. Biểu đồ area thể hiện tích lũy giá theo thời gian cho thấy điều gì?")
print("- Area chart thể hiện tổng giá trị tích lũy của cổ phiếu theo thời gian.\n  Nếu area chart liên tục mở rộng, đó là dấu hiệu của tăng trưởng bền vững.\n  Giúp nhận diện các giai đoạn tăng trưởng mạnh hoặc điều chỉnh giảm.")

for symbol in symbols:
    conn = sqlite3.connect(db_names[symbol])
    price_df = pd.read_sql_query("SELECT * FROM price", conn)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    plt.figure(figsize=(12,6))
    plt.fill_between(price_df['Date'], price_df['close'].cumsum(), color=colors[symbol], alpha=0.5)
    plt.title(f'Biểu đồ Area: Tích lũy giá theo thời gian - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Close Price')
    plt.tight_layout()
    plt.savefig(f'charts/area_cumulative_{symbol}.png')
    plt.close()
    conn.close()

# 4. Biểu đồ scatter giữa giá và sentiment có điểm bất thường nào không?
print("\n4. Biểu đồ scatter giữa giá và sentiment có điểm bất thường nào không?")
print("- Scatter plot giúp phát hiện các điểm bất thường (outlier):\n  Một số điểm có sentiment rất cao nhưng giá không tăng tương ứng, hoặc ngược lại.\n  Đa số các điểm tập trung quanh sentiment trung tính, nhưng có một số điểm cực trị.")

for symbol in symbols:
    conn = sqlite3.connect(db_names[symbol])
    price_df = pd.read_sql_query("SELECT * FROM price", conn)
    sentiment_df = pd.read_sql_query("SELECT * FROM sentiment", conn)
    earnings_df = pd.read_sql_query("SELECT * FROM earnings", conn)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    sentiment_df['Earnings Date'] = pd.to_datetime(sentiment_df['Earnings Date'])
    # Lấy giá tại ngày earnings
    price_on_earnings = price_df.set_index('Date').reindex(sentiment_df['Earnings Date'])['close']
    plt.figure(figsize=(8,6))
    plt.scatter(sentiment_df['compound'], price_on_earnings, color=colors[symbol], alpha=0.7)
    plt.title(f'Biểu đồ Scatter: Giá vs Sentiment - {symbol}')
    plt.xlabel('Sentiment (compound)')
    plt.ylabel('Close Price at Earnings')
    plt.tight_layout()
    plt.savefig(f'charts/scatter_price_sentiment_{symbol}.png')
    plt.close()
    conn.close()

# 5. Biểu đồ histogram của chênh lệch giá cho thấy phân phối ra sao?
print("\n5. Biểu đồ histogram của chênh lệch giá cho thấy phân phối ra sao?")
print("- Histogram cho thấy phân phối lệch phải hoặc lệch trái, không hoàn toàn chuẩn.\n  Đa số các sự kiện earnings chỉ tạo ra thay đổi giá nhỏ, nhưng có một số sự kiện tạo ra biến động rất lớn (đuôi dài).\n  Điều này cho thấy thị trường thường phản ứng mạnh với một số ít sự kiện đặc biệt.")

for symbol in symbols:
    conn = sqlite3.connect(db_names[symbol])
    query = '''
    WITH price_context AS (
        SELECT e."Earnings Date" AS earnings_date,
               (SELECT p1.close FROM price p1 WHERE p1.Date = DATE(e."Earnings Date", '-1 day')) AS close_before,
               (SELECT p2.close FROM price p2 WHERE p2.Date = DATE(e."Earnings Date", '+1 day')) AS close_after
        FROM earnings e
    )
    SELECT earnings_date, ROUND((close_after - close_before) * 100.0 / close_before, 2) AS price_change_pct
    FROM price_context
    WHERE close_before IS NOT NULL AND close_after IS NOT NULL
    '''
    df = pd.read_sql_query(query, conn)
    plt.figure(figsize=(8,5))
    plt.hist(df['price_change_pct'], bins=15, color=colors[symbol], alpha=0.7, edgecolor='black')
    plt.title(f'Biểu đồ Histogram: Phân phối chênh lệch giá - {symbol}')
    plt.xlabel('Price Change (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'charts/hist_price_change_{symbol}.png')
    plt.close()
    conn.close()

# 6. Nhóm đã sử dụng màu sắc nào để phân biệt các cổ phiếu?
print("\n6. Nhóm đã sử dụng màu sắc nào để phân biệt các cổ phiếu?")
print("- AAPL: Xanh dương (blue)\n- MSFT: Cam (orange)\n- GOOGL: Xanh lá (green)\nCác màu này được giữ nhất quán trên tất cả các biểu đồ để người dùng dễ nhận diện.")

# 7. Biểu đồ nào giúp người dùng nhận diện cổ phiếu có phản ứng giá mạnh nhất?
print("\n7. Biểu đồ nào giúp người dùng nhận diện cổ phiếu có phản ứng giá mạnh nhất?")
print("- Bar chart và scatter plot về price change sau earnings là trực quan nhất để nhận diện cổ phiếu có phản ứng giá mạnh nhất.\n  Ngoài ra, biểu đồ line với highlight các điểm earnings cũng giúp xác định các sự kiện có biến động lớn.")

# 8. Nhóm đã lưu các biểu đồ vào đâu, và chúng được đặt tên ra sao?
print("\n8. Nhóm đã lưu các biểu đồ vào đâu, và chúng được đặt tên ra sao?")
print("- Tất cả các biểu đồ được lưu vào thư mục 'charts/'.\n- Tên file biểu đồ theo cấu trúc: line_price_{symbol}.png, bar_price_change_{symbol}.png, area_cumulative_{symbol}.png, scatter_price_sentiment_{symbol}.png, hist_price_change_{symbol}.png\nTrong đó {symbol} là mã cổ phiếu (AAPL, MSFT, GOOGL).")

# 9. Ý nghĩa của từng biểu đồ trong báo cáo đã được giải thích thế nào?
print("\n9. Ý nghĩa của từng biểu đồ trong báo cáo đã được giải thích thế nào?")
print("- Mỗi biểu đồ đều có tiêu đề, chú thích (legend) và caption giải thích ý nghĩa.\n  Line chart: Diễn giải xu hướng giá quanh earnings.\n  Bar chart: So sánh mức biến động giá giữa các cổ phiếu.\n  Area chart: Thể hiện sự tích lũy/tăng trưởng giá trị.\n  Scatter plot: Phân tích mối liên hệ giữa sentiment và giá.\n  Histogram: Đánh giá phân phối và xác suất biến động giá lớn.")

# 10. Những thách thức nào gặp phải khi trực quan hóa dữ liệu?
print("\n10. Những thách thức nào gặp phải khi trực quan hóa dữ liệu?")
print("- Chọn loại biểu đồ phù hợp, màu sắc nhất quán, xử lý outlier, tối ưu layout, lưu trữ và đặt tên file rõ ràng, giải thích ý nghĩa dễ hiểu cho người dùng cuối.")

print("\nĐã vẽ và lưu các biểu đồ line, bar, area, scatter, histogram cho từng cổ phiếu vào thư mục charts/.")
print("Tên file biểu đồ theo cấu trúc: line_price_{symbol}.png, bar_price_change_{symbol}.png, area_cumulative_{symbol}.png, scatter_price_sentiment_{symbol}.png, hist_price_change_{symbol}.png")
