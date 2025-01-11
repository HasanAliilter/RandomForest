from src.preprocess import preprocess_data_in_chunks, load_multiple_files_in_chunks
from src.train_model import grid_search_random_forest
from src.evaluate_model import evaluate_model
from src.visualize import plot_feature_importance
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

data_directory = "Data"
chunk_size = 100000  # Daha küçük chunk boyutu

# Veriyi parçalara ayırarak yükleme ve işleme
data_chunks = load_multiple_files_in_chunks(data_directory, chunk_size)

processed_X_chunks = []
processed_y_chunks = []

for chunk in data_chunks:
    result = preprocess_data_in_chunks(chunk)
    if result is not None:
        X_resampled, y_resampled = result  # Hem X hem de y'yi alıyoruz
        processed_X_chunks.append(X_resampled)
        processed_y_chunks.append(y_resampled)

# Verileri birleştirme işlemi
if processed_X_chunks:  # Liste boş değilse, verileri birleştir
    X = pd.concat(processed_X_chunks, ignore_index=True)
    y = pd.concat(processed_y_chunks, ignore_index=True)

print(X.head())
print("Veri setindeki sütunlar:", X.columns.tolist())

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []
classification_reports = []

for train_index, test_index in kf.split(X):
    print(f"Fold {fold}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Model eğitimi (GridSearchCV ile hiperparametre optimizasyonu)
    model, X_test_pca = grid_search_random_forest(X_train, y_train, X_test)  # Burada GridSearch kullanıyoruz

    # Model değerlendirmesi
    accuracy, report = evaluate_model(model, X_test_pca, y_test)
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
