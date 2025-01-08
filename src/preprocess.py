import os
import pandas as pd
import numpy as np

def load_multiple_files_in_chunks(directory, chunk_size=500000):
    """
    Belirtilen dizindeki tüm CSV dosyalarını parçalara ayırarak yükler ve birleştirir.
    """
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"{directory} içinde CSV dosyası bulunamadı.")
    
    print(f"{len(all_files)} dosya bulundu. Dosyalar: {all_files}")
    data_frames = []
    
    for file in all_files:
        # Dosyayı parçalara ayırarak oku
        for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
            data_frames.append(chunk)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    print(f"Toplam birleştirilen satır sayısı: {combined_data.shape[0]}")
    return combined_data

def preprocess_data_in_chunks(data, chunk_size=500000):
    """
    Veriyi parçalara ayırarak işler.
    """
    if 'Label' in data.columns:
        label = data['Label']
        data = data.drop(columns=['Label'])
    else:
        raise KeyError("Veri setinde 'Label' sütunu bulunamadı.")
    
    numeric_data = data.select_dtypes(include=[np.number])
    
    print("Veri setinde NaN değer sayısı:", numeric_data.isna().sum().sum())
    print("Veri setinde sonsuz değer sayısı:", np.isinf(numeric_data).sum().sum())
    
    # NaN ve sonsuz değerleri temizle
    numeric_data.fillna(0, inplace=True)
    numeric_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Temizlenmiş sayısal sütunları geri ekle
    data = data[numeric_data.columns]
    data['Label'] = label  # Etiket sütununu geri ekle
    
    return data
