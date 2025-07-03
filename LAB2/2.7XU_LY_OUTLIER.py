import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

files = ["AAPL_scaled.csv", "MSFT_scaled.csv", "GOOGL_scaled.csv"]
labels = ["AAPL", "MSFT", "GOOGL"]

for file, label in zip(files, labels):
    df = pd.read_csv(file)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['Unnamed: 4', 'Unnamed: 1', 'Unnamed: 5']])
    plt.title(f"Boxplot - phát hiện outliers ({label}, dữ liệu đã scale)")
    plt.xlabel('Open (4), Close (1), Volume (5)')
    plt.show() 