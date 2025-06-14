import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import time
from datetime import datetime

def train_model(X_train, y_train, X_test):
    print(f"\nEğitim başlangıç zamanı: {datetime.now().strftime('%H:%M:%S')}")
    başlangıç_zamanı = time.time()
    
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())

    # Feature Selection için ilk RandomForest eğitimi
    print("\nÖzellik seçimi için ön model eğitiliyor...")
    özellik_seçim_başlangıç = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    özellik_seçim_süre = time.time() - özellik_seçim_başlangıç
    print(f"Özellik seçimi süresi: {özellik_seçim_süre:.2f} saniye")

    # Özelliklerin önemini alma
    importances = rf.feature_importances_
    feature_names = X_train.columns
    
    # Önem sırasına göre sıralama
    indices = np.argsort(importances)[::-1]

    print("\nÖzelliklerin Önemi:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    # Toplam önemin %95'ini kapsayan özellikleri seçiyoruz
    cumulative_importance = np.cumsum(importances[indices])
    num_features_to_keep = np.where(cumulative_importance >= 0.95)[0][0] + 1
    selected_features = indices[:num_features_to_keep]
    
    print(f"\nSeçilen özellik sayısı: {num_features_to_keep}")
    
    # Seçilen özelliklerle X_train ve X_test'i güncelleme
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    # RandomizedSearchCV için parametre dağılımları
    param_distributions = {
        'n_estimators': [50],
        'max_depth': [ 20, ],
        'min_samples_split': [2],
        'min_samples_leaf': [ 2],
        'class_weight': ['balanced']
    }

    rf_model = RandomForestClassifier(random_state=42)

    print("\nHiperparametre optimizasyonu başlıyor...")
    optimizasyon_başlangıç = time.time()
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train_selected, y_train)
    optimizasyon_süre = time.time() - optimizasyon_başlangıç
    print(f"Hiperparametre optimizasyonu süresi: {optimizasyon_süre:.2f} saniye")

    # RandomizedSearchCV sonuçlarını ekrana yazdırma
    print("\nRandomizedSearchCV Sonuçları:")
    print("En iyi parametreler:", random_search.best_params_)
    print(f"En iyi çapraz doğrulama skoru: {random_search.best_score_:.4f}")

    # Test tahminleri için süre ölçümü
    print("\nTest seti üzerinde tahmin yapılıyor...")
    tahmin_başlangıç = time.time()
    test_tahminleri = random_search.predict(X_test_selected)
    tahmin_süre = time.time() - tahmin_başlangıç
    print(f"Test tahminleri süresi: {tahmin_süre:.2f} saniye")

    toplam_süre = time.time() - başlangıç_zamanı
    print(f"\nToplam işlem süresi: {toplam_süre:.2f} saniye")
    print(f"Eğitim bitiş zamanı: {datetime.now().strftime('%H:%M:%S')}")

    # Süre özeti
    print("\nSüre Özeti:")
    print(f"1. Özellik seçimi: {özellik_seçim_süre:.2f} saniye")
    print(f"2. Hiperparametre optimizasyonu: {optimizasyon_süre:.2f} saniye")
    print(f"3. Test tahminleri: {tahmin_süre:.2f} saniye")
    print(f"4. Toplam süre: {toplam_süre:.2f} saniye")

    return random_search.best_estimator_, X_test_selected
