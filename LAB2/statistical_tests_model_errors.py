import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from itertools import combinations

RESULTS_DIR = 'results'
stocks = ['AAPL', 'MSFT', 'GOOGL']

for stock in stocks:
    print(f'\n=== {stock} ===')
    # Đọc lỗi dự báo từ file pipeline (nếu đã lưu), hoặc tính lại từ y_test, y_pred
    model_results_file = os.path.join(RESULTS_DIR, f'{stock}_model_results.csv')
    if not os.path.exists(model_results_file):
        print(f'Không tìm thấy file {model_results_file}')
        continue
    # Đọc file pipeline lưu lại (chỉ có MAE, RMSE, R2, BestParams, BestSearch)
    # Để lấy lỗi từng mô hình, cần đọc lại y_test, y_pred từ script pipeline (nên lưu riêng file npy hoặc pickle)
    # Ở đây giả sử bạn đã lưu y_test, y_pred cho từng mô hình vào file npy
    errors = []
    model_names = []
    for i in range(7):
        err_file = os.path.join(RESULTS_DIR, f'{stock}_model_{i}_errors.npy')
        if os.path.exists(err_file):
            errors.append(np.load(err_file))
        else:
            print(f'Không tìm thấy file lỗi: {err_file}')
    # Nếu chưa có file lỗi, bỏ qua kiểm định
    if len(errors) < 2:
        print('Không đủ dữ liệu lỗi để kiểm định.')
        continue
    # Đọc tên mô hình
    df = pd.read_csv(model_results_file)
    model_names = df['Model'].tolist()
    # ANOVA
    f_stat, p_value = f_oneway(*errors)
    print(f'ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4g}')
    if p_value < 0.05:
        print('=> Có sự khác biệt có ý nghĩa thống kê giữa các mô hình.')
    else:
        print('=> Không có sự khác biệt rõ rệt giữa các mô hình.')
    # T-test từng cặp mô hình
    for (i, j) in combinations(range(len(errors)), 2):
        t_stat, t_p = ttest_ind(errors[i], errors[j])
        print(f'T-test giữa {model_names[i]} và {model_names[j]}: p-value = {t_p:.4g}') 