import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, X_test):
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())

    # Feature Selection için XGBoost ile model eğitimi
    print("Özellik seçimi için XGBoost modeli eğitiliyor...")
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)

    # Özelliklerin önemini alma
    importances = xgb.feature_importances_
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

    # GridSearchCV için parametreler
    param_grid = {
        'n_estimators': [50],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [2],
        'class_weight': ['balanced']
    }

    rf_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)

    # GridSearchCV sonuçlarını ekrana yazdırma
    print("\nGridSearchCV Sonuçları:")
    print("En iyi parametreler:", grid_search.best_params_)

    # En iyi modeli ve test setini döndürüyoruz
    return grid_search.best_estimator_, X_test_selected
