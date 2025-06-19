import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

def train_model(X_train, y_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())

    # Feature Selection için Logistic Regression ile ilk model eğitimi
    print("\nÖzellik seçimi için Logistic Regression modeli eğitiliyor...")
    lr = LogisticRegression(
    penalty='l2', solver='saga', max_iter=1000,
    multi_class='multinomial', random_state=42
)
    lr.fit(X_train, y_train)

    importances = np.abs(lr.coef_[0])
    feature_names = X_train.columns
    
    indices = np.argsort(importances)[::-1]

    cumulative_importance = np.cumsum(importances[indices])
    total_importance = cumulative_importance[-1]
    num_features_to_keep = np.where(cumulative_importance >= 0.95 * total_importance)[0][0] + 1
    selected_features = indices[:num_features_to_keep]
    
    print(f"\nSeçilen özellik sayısı: {num_features_to_keep}")
    
    # Seçilen özelliklerle X_train ve X_test'i güncelleme
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

    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist')

    # GridSearchCV başlatma
    print("\nGridSearchCV ile XGBoost modeli eğitiliyor...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)

    # En iyi modeli ve test setini döndürüyoruz
    return grid_search.best_estimator_, X_test_selected