import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('results', exist_ok=True)
stocks = ['AAPL', 'MSFT', 'GOOGL']
file_map = {s: f'{s}_feature.csv' for s in stocks}

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
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c or c == 'Date'])
    df = df.dropna()
    X = df.drop(columns=['pct_change'])
    y = df['pct_change']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    stock_results = []
    for name, (model, params) in models.items():
        best_mae = None
        best_model = None
        best_params = None
        best_search = None
        best_y_pred = None
        # Nếu có params, thử cả GridSearchCV và RandomizedSearchCV
        if params:
            grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            grid_mae = -grid.best_score_
            grid_model = grid.best_estimator_
            grid_params = grid.best_params_
            rand = RandomizedSearchCV(model, params, n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
            rand.fit(X_train, y_train)
            rand_mae = -rand.best_score_
            rand_model = rand.best_estimator_
            rand_params = rand.best_params_
            # Chọn search tốt nhất
            if grid_mae <= rand_mae:
                best_mae = grid_mae
                best_model = grid_model
                best_params = grid_params
                best_search = 'GridSearchCV'
            else:
                best_mae = rand_mae
                best_model = rand_model
                best_params = rand_params
                best_search = 'RandomizedSearchCV'
        else:
            best_model = model.fit(X_train, y_train)
            best_params = {}
            best_search = 'None'
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # Lưu lỗi dự báo ra file npy
        error = y_test.values - y_pred
        np.save(f'results/{stock}_model_{len(stock_results)}_errors.npy', error)
        stock_results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'BestParams': best_params,
            'BestSearch': best_search,
            'y_test': y_test.values,
            'y_pred': y_pred
        })
        print(f'{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, BestParams={best_params}, BestSearch={best_search}')
    results_summary.append({'Stock': stock, 'Results': stock_results})
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
    errors = [r['y_test'] - r['y_pred'] for r in stock_results]
    plt.figure(figsize=(10,5))
    sns.boxplot(data=errors)
    plt.xticks(ticks=range(len(model_names)), labels=model_names)
    plt.title(f'{stock} - Error Distribution by Model')
    plt.ylabel('Error (y_true - y_pred)')
    plt.savefig(f'results/{stock}_error_boxplot.png')
    plt.close()
    best_idx = np.argmin(maes)
    best = stock_results[best_idx]
    plt.figure(figsize=(12,6))
    plt.plot(best['y_test'], label='y_true', marker='o')
    plt.plot(best['y_pred'], label='y_pred', marker='x')
    plt.title(f'{stock} - {best["Model"]} (Best MAE) - y_true vs y_pred')
    plt.legend()
    plt.savefig(f'results/{stock}_best_model_line.png')
    plt.close()
    pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_pred']} for r in stock_results]).to_csv(f'results/{stock}_model_results.csv', index=False)

with open('results/summary.txt', 'w', encoding='utf-8') as f:
    for res in results_summary:
        stock = res['Stock']
        f.write(f'=== {stock} ===\n')
        for r in res['Results']:
            f.write(f"{r['Model']}: MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, R2={r['R2']:.4f}, BestParams={r['BestParams']}, BestSearch={r['BestSearch']}\n")
        f.write('\n')
print('Hoàn thành pipeline. Kết quả lưu ở thư mục results/.') 