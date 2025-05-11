from src.preprocess import preprocess_data_in_chunks, load_multiple_files_in_chunks
from src.train_model_options.fs_pca_tm_random_forest94 import train_model
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
        processed_X_chunks.append(X_resampled) #X_resampled: Özellik (feature) verilerini
        processed_y_chunks.append(y_resampled) #y_resampled: Etiket (label) verilerini tutmak için kullanılır.

# Verileri birleştirme işlemi tüm chunklanmış veriler birleştirilir
if processed_X_chunks:  # Liste boş değilse, verileri birleştir
    X = pd.concat(processed_X_chunks, ignore_index=True)
    y = pd.concat(processed_y_chunks, ignore_index=True)

print(X.head())
print("Veri setindeki sütunlar:", X.columns.tolist()) #kontrol amaçlı bir print

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = [] #Her fold’un doğruluk değerini saklar.
classification_reports = [] #Her fold için sınıflandırma metriklerini (precision, recall, f1-score) saklar.

for train_index, test_index in kf.split(X): #kf.split(X): Veriyi eğitim ve test indekslerine böler. train_index ve test_index: Fold’a özel indeks dizileri döner.
    print(f"Fold {fold}")
    #Eğitim (X_train, y_train) ve test (X_test, y_test) verileri iloc ile indekslere göre seçilir. X ve y, pandas DataFrame veya Series’tir.
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Model eğitimi (GridSearchCV ile hiperparametre optimizasyonu)
    model, X_test_pca = train_model(X_train, y_train, X_test)  # Burada GridSearch kullanıyoruz

    # Model değerlendirmesi
    accuracy, report = evaluate_model(model, X_test_pca, y_test)
    accuracies.append(accuracy)
    classification_reports.append(report)

    print(f"Fold {fold} Accuracy:", accuracy)
    print(f"Fold {fold} Classification Report:\n", report)
    fold += 1

# Ortalama doğruluk ve rapor
mean_accuracy = np.mean(accuracies)
print("\nOrtalama Accuracy:", mean_accuracy) #doğruluk değerini ekrana yazdırır

# Özellik önemini görselleştirme (son katmanda eğitilen model üzerinden)
plot_feature_importance(model, X.columns)

# Sonuçları dosyaya yazma
with open('results/model_performance.txt', 'w') as f: #ortalama doğruluğu dosyaya yazmak için dosyayı açar ve yoksa oluşturur
    f.write(f"Ortalama Accuracy: {mean_accuracy}\n\n")
    for i, report in enumerate(classification_reports, start=1):
        f.write(f"Fold {i} Classification Report:\n{report}\n\n")
