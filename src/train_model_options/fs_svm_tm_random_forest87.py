from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd

def train_model(X_train, y_train, X_test):
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    print("NaN değerler dolduruldu.")

    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Özellik seçimi için LinearSVC modeli
    print("Özellik seçimi için LinearSVC modeli eğitiliyor...")
    svc = LinearSVC(C=0.01, penalty="l1", dual=False, random_state=42, max_iter=100000)
    svc.fit(X_train, y_train)

    importances = np.abs(svc.coef_).flatten()
    indices = np.argsort(importances)[::-1]

    cumulative_importance = np.cumsum(importances[indices])
    num_features_to_keep = np.where(cumulative_importance >= 0.95)[0][0] + 1
    selected_features = indices[:num_features_to_keep]
    
    print(f"\nSeçilen özellik sayısı: {num_features_to_keep}")
    
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

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

    return grid_search.best_estimator_, X_test_selected
