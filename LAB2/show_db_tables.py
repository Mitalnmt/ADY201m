import sqlite3
import pandas as pd

dbs = ['aapl_analysis.db', 'msft_analysis.db', 'googl_analysis.db']
for db in dbs:
    print(f"\n--- Hiển thị dữ liệu trong DB: {db} ---")
    conn = sqlite3.connect(db)
    for table in ['price', 'earnings', 'sentiment']:
        print(f"\nBảng: {table}")
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", conn)
            print(df)
        except Exception as e:
            print(f"Lỗi khi đọc bảng {table}: {e}")
    conn.close() 