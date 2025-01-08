from src.preprocess import preprocess_data_in_chunks, load_multiple_files_in_chunks
from src.train_model import train_random_forest_in_chunks
from src.evaluate_model import evaluate_model
from src.visualize import plot_feature_importance
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

data_directory = "Data"
data = load_multiple_files_in_chunks(data_directory)  # Veriyi parçalara ayırarak yükleme
data = preprocess_data_in_chunks(data)  # Veriyi parçalara ayırarak işleme

print(data.head())
print("Veri setindeki sütunlar:", data.columns.tolist())

if 'Label' not in data.columns:
    raise KeyError("Veri setinde 'Label' sütunu bulunamadı. Lütfen veri setini kontrol edin.")

X = data.drop(columns=['Label'])
y = data['Label']

def clean_data(X, y):
    """NaN ve sonsuz değerleri temizleme."""
    print("NaN değer sayısı (öncesi):", X.isnull().sum().sum())
    print("Sonsuz değer sayısı (öncesi):", np.isinf(X).sum().sum())

    X.fillna(X.mean(), inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    y = y[X.index]
    print("y NaN değer sayısı (sonrası):", y.isna().sum())

    print("NaN değer sayısı (sonrası):", X.isnull().sum().sum())
    print("Sonsuz değer sayısı (sonrası):", np.isinf(X).sum().sum())

    return X, y

X, y = clean_data(X, y)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []
classification_reports = []

for train_index, test_index in kf.split(X):
    print(f"Fold {fold}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Model eğitimi
    model = train_random_forest_in_chunks(X_train, y_train)

    # Model değerlendirmesi
    accuracy, report = evaluate_model(model, X_test, y_test)
    accuracies.append(accuracy)
    classification_reports.append(report)

    print(f"Fold {fold} Accuracy:", accuracy)
    print(f"Fold {fold} Classification Report:\n", report)
    fold += 1

# Ortalama doğruluk ve rapor
mean_accuracy = np.mean(accuracies)
print("\nOrtalama Accuracy:", mean_accuracy)

# Özellik önemini görselleştirme (son katmanda eğitilen model üzerinden)
plot_feature_importance(model, X.columns)

# Sonuçları dosyaya yazma
with open('results/model_performance.txt', 'w') as f:
    f.write(f"Ortalama Accuracy: {mean_accuracy}\n\n")
    for i, report in enumerate(classification_reports, start=1):
        f.write(f"Fold {i} Classification Report:\n{report}\n\n")
