from src.preprocess import preprocess_data_in_chunks, load_multiple_files_in_chunks
from src.train_model_options.fs_pca_tm_random_forest94 import train_model
from src.evaluate_model import evaluate_model
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import time
from datetime import datetime

start_time = datetime.now()
print(f"başlangıç zamanı: {start_time.strftime('%H:%M:%S')}")

preprocess_start = time.time()
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

preprocess_end = time.time()
preprocess_time = preprocess_end - preprocess_start
print(f"1. preprocess süresi: {preprocess_time:.2f} saniye")
# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []
classification_reports = []

for train_index, test_index in kf.split(X):
    print(f"Fold {fold}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Zaman ölçümleri
    feature_start = time.time()

    # Model eğitimi (GridSearchCV ile hiperparametre optimizasyonu)
    model, X_test_pca = train_model(X_train, y_train, X_test)

    feature_end = time.time()
    feature_selection_time = feature_end - feature_start

    # Model değerlendirmesi
    prediction_start = time.time()
    accuracy, report = evaluate_model(model, X_test_pca, y_test)
    prediction_end = time.time()
    prediction_time = prediction_end - prediction_start
    total_time = feature_selection_time + prediction_time
    print(f"\nSüre Özeti:")
    print(f"2. Özellik seçimi + hiperparametre optimizasyonu: {feature_selection_time:.2f} saniye")
    print(f"3. Test tahminleri: {prediction_time:.2f} saniye")
    print(f"4. fold tamamlama süresi: {total_time:.2f} saniye")
    accuracies.append(accuracy)
    classification_reports.append(report)

    print(f"Fold {fold} Accuracy:", accuracy)
    print(f"Fold {fold} Classification Report:\n", report)
    fold += 1

# Ortalama doğruluk ve rapor
mean_accuracy = np.mean(accuracies)
print("\nOrtalama Accuracy:", mean_accuracy)

# Sonuçları dosyaya yazma
with open('results/model_performance.txt', 'w') as f:
    f.write(f"Ortalama Accuracy: {mean_accuracy}\n\n")
    for i, report in enumerate(classification_reports, start=1):
        f.write(f"Fold {i} Classification Report:\n{report}\n\n")

end_time = datetime.now()
print(f"\nEğitim bitiş zamanı: {end_time.strftime('%H:%M:%S')}")

