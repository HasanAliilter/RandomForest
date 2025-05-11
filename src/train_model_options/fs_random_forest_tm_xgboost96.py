import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train, X_test):
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum()) #X_train içindeki NaN (eksik) değerlerin toplam sayısını verir.
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum()) #X_train içindeki sonsuz (inf) değerlerin toplam sayısını verir.

    # Feature Selection için RandomForest ile ilk model eğitimi
    print("\nÖzellik seçimi için RandomForest modeli eğitiliyor...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Özelliklerin önemini alma
    importances = rf.feature_importances_ # Her özelliğin modele katkısını gösteren önem skorlarını verir.
    feature_names = X_train.columns 
    
    # Önem sırasına göre sıralama
    indices = np.argsort(importances)[::-1] #Özelliklerin önem sırasına göre indeksleri. Sıralama indekslerini döndürür. [::-1] ile büyükten küçüğe sıralanır.

    print("\nÖzelliklerin Önemi:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    # Toplam önemin %95'ini kapsayan özellikleri seçiyoruz
    cumulative_importance = np.cumsum(importances[indices]) #np.cumsum(...): Kümülatif toplam hesaplar.
    num_features_to_keep = np.where(cumulative_importance >= 0.95)[0][0] + 1 #np.where(...): İlk kez %95'ten büyük olan konumu bulur.
    selected_features = indices[:num_features_to_keep] # Böylece, toplam bilgi içeriğinin %95’ini sağlayan en önemli ilk n özelliği seçiyoruz.
    
    print(f"\nSeçilen özellik sayısı: {num_features_to_keep}")
    
    # Seçilen özelliklerle X_train ve X_test'i güncelleme
    #iloc[:, selected_features]: Belirlenen özellik indekslerine göre sadece bu sütunları alır.
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    # GridSearchCV için parametreler
    param_grid = {
        'n_estimators': [50],        # Ağaç sayısı
        'max_depth': [20],           # Maksimum derinlik
        'learning_rate': [0.01],     # Öğrenme oranı
        'gamma': [1],                # Bölünme kontrolü
        'subsample': [0.8]           # Örnekleme oranı
    }

    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist', device="cuda")

    # GridSearchCV başlatma
    print("\nGridSearchCV ile XGBoost modeli eğitiliyor...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)

    # GridSearchCV sonuçlarını ekrana yazdırma
    print("\nGridSearchCV Sonuçları:")
    print("En iyi parametreler:", grid_search.best_params_)

    # En iyi modeli ve test setini döndürüyoruz
    return grid_search.best_estimator_, X_test_selected
