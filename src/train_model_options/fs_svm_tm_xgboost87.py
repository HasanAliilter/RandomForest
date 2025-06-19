from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
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

    # Özellik seçimi için LinearSVC (penalty='l2') ve SelectFromModel kullanımı
    print("Özellik seçimi için LinearSVC modeli eğitiliyor...")
    
    svc = LinearSVC(C=0.01, penalty="l2", dual=True, random_state=42, max_iter=5000)
    svc.fit(X_train, y_train)

    selector_model = SelectFromModel(svc, prefit=True, threshold="median")
    X_train_selected = selector_model.transform(X_train)
    X_test_selected = selector_model.transform(X_test)

    print(f"Seçilen özellik sayısı: {X_train_selected.shape[1]}")

    # GridSearchCV için parametreler (XGBoost için)
    param_grid = {
        'n_estimators': [50],
        'max_depth': [20],
        'learning_rate': [0.01],
        'gamma': [1],
        'subsample': [0.8]
    }

    # XGBoost modeli tanımlama
    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42,
                              use_label_encoder=False, eval_metric='logloss', tree_method='hist')

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)

    return grid_search.best_estimator_, X_test_selected
