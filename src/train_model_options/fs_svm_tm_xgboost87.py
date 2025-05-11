from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd

def train_model(X_train, y_train, X_test):
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())

    # NaN değerleri median ile doldur
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    print("NaN değerler dolduruldu.")

    # Düşük varyanslı özellikleri kaldır
    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    # Özellikleri ölçeklendirme
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Özellik seçimi için LinearSVC modeli
    print("Özellik seçimi için LinearSVC modeli eğitiliyor...")
    svc = LinearSVC(C=0.01, penalty="l1", dual=False, random_state=42, max_iter=100000)
    svc.fit(X_train, y_train)

    importances = np.abs(svc.coef_).flatten()
    indices = np.argsort(importances)[::-1]

    # %95 önem kapsayan özellikleri seç
    cumulative_importance = np.cumsum(importances[indices])
    num_features_to_keep = np.where(cumulative_importance >= 0.95)[0][0] + 1
    selected_features = indices[:num_features_to_keep]
    
    print(f"\nSeçilen özellik sayısı: {num_features_to_keep}")
    
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # GridSearchCV için parametreler (XGBoost için)
    param_grid = {
        'n_estimators': [50],        # Ağaç sayısı
        'max_depth': [20],           # Maksimum derinlik
        'learning_rate': [0.01],     # Öğrenme oranı
        'gamma': [1],                # Bölünme kontrolü
        'subsample': [0.8]           # Örnekleme oranı
    }

    # XGBoost modeli tanımlama
    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)

    print("\nGridSearchCV Sonuçları:")
    print("En iyi parametreler:", grid_search.best_params_)

    return grid_search.best_estimator_, X_test_selected
