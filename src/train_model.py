import numpy as np
from sklearn.calibration import LabelEncoder
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import time

def grid_search_xgboost(X_train, y_train, X_test):
    print("BURDAAAAAAAAAAAA")
    #print("GPU Kullanılabilirliği:", rabit.get_processor_info())
    start_gpu = time.time()
  
    label_encoder= LabelEncoder()

    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())
    
    print("y_train tipi:", type(y_train))
    
    if isinstance(y_train, pd.Series):
        print("y_train NaN değer sayısı:", y_train.isna().sum())
        y_train = y_train.dropna() 
        X_train = X_train.loc[y_train.index] 
    else:
        print("y_train NaN değer sayısı:", np.isnan(y_train).sum())

    # NaN değerlerini doldurmak için SimpleImputer kullanıyoruz
    print("X_train'deki NaN değerler dolduruluyor...")
    imputer = SimpleImputer(strategy='mean')  # 'mean' yerine 'median' da kullanabilirsiniz
    X_train_imputed = imputer.fit_transform(X_train)  # NaN'leri X_train için dolduruyoruz
    X_test_imputed = imputer.transform(X_test)  # NaN'leri X_test için de dolduruyoruz
    
    # SMOTE ile veri dengeleme
    print("SMOTE ile veri dengeleme uygulanıyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

    # PCA uygulama (isteğe bağlı, daha fazla bileşen kullanılarak veri kaybı minimize edilebilir)
    pca = PCA(n_components=0.98)  # Variance'ı %98 tutacak şekilde PCA
    X_train_resampled = pca.fit_transform(X_train_resampled)
    X_test_pca = pca.transform(X_test_imputed)  # Test verisini de PCA ile dönüştür

    print(f"PCA sonrası yeni özellik sayısı: {X_train_resampled.shape[1]}")

    # GridSearchCV için parametre grid'i (daha dar bir grid kullanıyoruz)
    param_grid = {
    'n_estimators': [50, 100],  # Ağaç sayısı
    'max_depth': [10, 20],  # Ağaç derinliği
    'learning_rate': [0.01, 0.1],  # Öğrenme oranı
    'gamma': [0, 1],  # Ağaç bölünmelerini kontrol eder
    'subsample': [0.8, 1.0],  # Örnekleme oranı
}

    # XGBoost modelini başlatma
    
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss',tree_method='hist',device="cuda")

    # GridSearchCV başlatma
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    # GridSearchCV ile model eğitimi
    grid_search.fit(X_train_resampled, y_train_resampled, ) #verbose = 'true'
    end_gpu = time.time()
    print(f"GPU Eğitim Süresi: {end_gpu - start_gpu:.2f} saniye")
    # En iyi model parametreleri
    print("En iyi parametreler:", grid_search.best_params_)
    print(label_encoder.classes_)

    # En iyi model döndürülüyor
    return grid_search.best_estimator_, X_test_pca
