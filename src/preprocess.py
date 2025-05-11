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

#Veri çerçevesindeki büyük veri tiplerini float64, int64 daha küçük ve daha verimli olanlara (float32, int32) çevirerek bellek kullanımını azaltır.
def optimize_memory(df):
    """
    Bellek optimizasyonu için veri tiplerini küçültür.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

#Veri setindeki NaN (eksik) ve sonsuz (inf) değerleri temizlenir ve nan ve inf değerler konsola yazılır.
def clean_data(chunk, fill_strategy='zero'):
    """
    NaN ve sonsuz değerleri temizler.
    fill_strategy:
        - 'zero': NaN ve inf değerlerini 0 ile doldurur
        - 'mean': Sadece inf değerlerini 0 yapar, NaN'leri ortalama ile doldurur (SimpleImputer ile)
    """
    numeric_data = chunk.select_dtypes(include=[np.number])
    print("NaN değer sayısı (öncesi):", numeric_data.isna().sum().sum())
    print("Sonsuz değer sayısı (öncesi):", np.isinf(numeric_data).sum().sum())

    # Sonsuz değerleri sıfırla
    numeric_data.replace([np.inf, -np.inf], 0, inplace=True)

    if fill_strategy == 'zero':
        numeric_data.fillna(0, inplace=True)
    elif fill_strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
        numeric_data[:] = imputer.fit_transform(numeric_data)
    else:
        raise ValueError("Geçersiz fill_strategy. 'zero' veya 'mean' olmalı.")

    print("NaN değer sayısı (sonrası):", numeric_data.isna().sum().sum())
    print("Sonsuz değer sayısı (sonrası):", np.isinf(numeric_data).sum().sum())
    return numeric_data


def preprocess_data_in_chunks(chunk, fill_strategy='zero'):
    """
    Veriyi parçalara ayırarak işler.
    fill_strategy:
        - 'zero': NaN ve inf değerlerini 0 ile doldurur
        - 'mean': Sadece inf değerlerini 0 yapar, NaN'leri ortalama ile doldurur
    """
    if 'Label' not in chunk.columns:
        raise KeyError("Veri setinde 'Label' sütunu bulunamadı.")
    
    # Sayısal veriyi seç
    numeric_data = chunk.select_dtypes(include=['float64', 'int64'])
    
    # Sayısal veri olup olmadığını kontrol et
    if numeric_data.empty:
        print("Uyarı: Sayısal veri bulunamadı. Lütfen veri setinizi kontrol edin.")
        return None
    
    # Bellek optimizasyonu ve veri temizleme
    numeric_data = optimize_memory(numeric_data)
    numeric_data = clean_data(numeric_data, fill_strategy=fill_strategy)
    
    # Etiket sütununu ekle ve dönüştür
    numeric_data['Label'] = chunk['Label']
    label_encoder = LabelEncoder()
    numeric_data['Label'] = label_encoder.fit_transform(numeric_data['Label'])

    # SMOTE ile sınıf dengesizliğini gider
    if numeric_data['Label'].nunique() > 1:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(
            numeric_data.drop('Label', axis=1), 
            numeric_data['Label']
        )
        return X_resampled, y_resampled
    else:
        print("Sadece bir sınıf bulundu. SMOTE işlemi yapılamaz.")
        return None
