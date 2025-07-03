import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo thư mục lưu kết quả
os.makedirs('results', exist_ok=True)

# Danh sách mã cổ phiếu và file dữ liệu
stocks = ['AAPL', 'MSFT', 'GOOGL']
file_map = {s: f'{s}_feature.csv' for s in stocks}

# Các mô hình và tham số tinh chỉnh
models = {
    'LinearRegression': (LinearRegression(), {}),
    'DecisionTree': (DecisionTreeRegressor(random_state=42), {'max_depth': [3, 5, 7, 10, None]}),
    'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 10]}),
    'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    'Lasso': (Lasso(max_iter=10000), {'alpha': [0.01, 0.1, 1.0]}),
    'RandomForest': (RandomForestRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [3, 5, 7, None]}),
    'SVR': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']})
}

results_summary = []

for stock, file in file_map.items():
    print(f'\n=== {stock} ===')
    df = pd.read_csv(file)
    # Loại bỏ các cột không dùng
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c or c == 'Date'])
    df = df.dropna()
    # Đặc trưng và biến mục tiêu
    X = df.drop(columns=['pct_change'])
    y = df['pct_change']
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    stock_results = []
    for name, (model, params) in models.items():
        if params:
            grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            best_model = model.fit(X_train, y_train)
            best_params = {}
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        stock_results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'BestParams': best_params,
            'y_test': y_test.values,
            'y_pred': y_pred
        })
        print(f'{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, BestParams={best_params}')
    # Lưu kết quả tổng hợp
    results_summary.append({'Stock': stock, 'Results': stock_results})
    # Vẽ bar chart MAE, RMSE
    maes = [r['MAE'] for r in stock_results]
    rmses = [r['RMSE'] for r in stock_results]
    model_names = [r['Model'] for r in stock_results]
    plt.figure(figsize=(10,5))
    sns.barplot(x=model_names, y=maes)
    plt.title(f'{stock} - MAE by Model')
    plt.ylabel('MAE')
    plt.savefig(f'results/{stock}_mae_bar.png')
    plt.close()
    plt.figure(figsize=(10,5))
    sns.barplot(x=model_names, y=rmses)
    plt.title(f'{stock} - RMSE by Model')
    plt.ylabel('RMSE')
    plt.savefig(f'results/{stock}_rmse_bar.png')
    plt.close()
    # Boxplot phân phối lỗi
    errors = [r['y_test'] - r['y_pred'] for r in stock_results]
    plt.figure(figsize=(10,5))
    sns.boxplot(data=errors)
    plt.xticks(ticks=range(len(model_names)), labels=model_names)
    plt.title(f'{stock} - Error Distribution by Model')
    plt.ylabel('Error (y_true - y_pred)')
    plt.savefig(f'results/{stock}_error_boxplot.png')
    plt.close()
    # Line chart y thực tế vs y dự báo cho mô hình tốt nhất
    best_idx = np.argmin(maes)
    best = stock_results[best_idx]
    plt.figure(figsize=(12,6))
    plt.plot(best['y_test'], label='y_true', marker='o')
    plt.plot(best['y_pred'], label='y_pred', marker='x')
    plt.title(f'{stock} - {best["Model"]} (Best MAE) - y_true vs y_pred')
    plt.legend()
    plt.savefig(f'results/{stock}_best_model_line.png')
    plt.close()
    # Lưu bảng kết quả
    pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_pred']} for r in stock_results]).to_csv(f'results/{stock}_model_results.csv', index=False)

# Tổng hợp báo cáo
with open('results/summary.txt', 'w', encoding='utf-8') as f:
    for res in results_summary:
        stock = res['Stock']
        f.write(f'=== {stock} ===\n')
        for r in res['Results']:
            f.write(f"{r['Model']}: MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, R2={r['R2']:.4f}, BestParams={r['BestParams']}\n")
        f.write('\n')
print('Hoàn thành pipeline. Kết quả lưu ở thư mục results/.') 