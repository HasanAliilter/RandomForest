import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

def load_multiple_files_in_chunks(directory, chunk_size=50000):
    """
    Belirtilen dizindeki tüm CSV dosyalarını parçalara ayırarak yükler.
    """
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"{directory} içinde CSV dosyası bulunamadı.")
    
    print(f"{len(all_files)} dosya bulundu. Dosyalar: {all_files}")
    
    for file in all_files:
        for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
            yield chunk

def optimize_memory(df):
    """
    Bellek optimizasyonu için veri tiplerini küçültür.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def clean_data(chunk):
    """
    NaN ve sonsuz değerleri temizler.
    """
    numeric_data = chunk.select_dtypes(include=[np.number])
    print("NaN değer sayısı (öncesi):", numeric_data.isna().sum().sum())
    print("Sonsuz değer sayısı (öncesi):", np.isinf(numeric_data).sum().sum())

    # NaN ve sonsuz değerleri işleme
    numeric_data.fillna(0, inplace=True)
    numeric_data.replace([np.inf, -np.inf], 0, inplace=True)

    print("NaN değer sayısı (sonrası):", numeric_data.isna().sum().sum())
    print("Sonsuz değer sayısı (sonrası):", np.isinf(numeric_data).sum().sum())
    return numeric_data

def preprocess_data_in_chunks(chunk):
    """
    Veriyi parçalara ayırarak işler.
    """
    if 'Label' not in chunk.columns:
        raise KeyError("Veri setinde 'Label' sütunu bulunamadı.")
    
    # Sayısal veriyi seçerken, sayısal olmayanları ayıklıyoruz
    numeric_data = chunk.select_dtypes(include=['float64', 'int64'])
    
    # Sayısal veri olup olmadığını kontrol et
    if numeric_data.empty:
        print("Uyarı: Sayısal veri bulunamadı. Lütfen veri setinizi kontrol edin.")
        return None
    
    # Bellek optimizasyonu
    numeric_data = optimize_memory(numeric_data)

    # NaN ve sonsuz değerleri temizleme
    numeric_data = clean_data(numeric_data)

    # NaN değerlerini doldurmak için SimpleImputer kullanıyoruz
    imputer = SimpleImputer(strategy='mean')  # İsterseniz 'median' da kullanabilirsiniz
    
    try:
        # NaN'leri dolduruyoruz
        numeric_data_imputed = imputer.fit_transform(numeric_data)
    except ValueError as e:
        print(f"Error during imputation: {e}")
        return None  # Hata durumunda None döndürebiliriz

    # Dönüştürülen numpy.ndarray'i tekrar DataFrame'e dönüştürme
    numeric_data_imputed = pd.DataFrame(numeric_data_imputed, columns=numeric_data.columns)

    # Etiket sütununu geri ekle
    numeric_data_imputed['Label'] = chunk['Label']

    # Label sütununu kategorik hale getirme
    label_encoder = LabelEncoder()
    numeric_data_imputed['Label'] = label_encoder.fit_transform(numeric_data_imputed['Label'])

    # Sınıf dağılımını kontrol et ve SMOTE işlemi yap
    if numeric_data_imputed['Label'].nunique() > 1:  # Eğer birden fazla sınıf varsa
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(
            numeric_data_imputed.drop('Label', axis=1), 
            numeric_data_imputed['Label']
        )
        # Hem X hem de y döndürüyoruz
        return X_resampled, y_resampled
    else:
        print("Sadece bir sınıf bulundu. SMOTE işlemi yapılamaz.")
        return None
