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
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')] #data klasörü içindeki tüm .csv uzantılı dosyaları listeye alır.
    if not all_files:
        raise FileNotFoundError(f"{directory} içinde CSV dosyası bulunamadı.")
    
    print(f"{len(all_files)} dosya bulundu. Dosyalar: {all_files}")
    
    for file in all_files: # Her dosya, belirtilen chunk_size kadar parçalara ayrılarak sırayla döndürülür.
        for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
            yield chunk # yield kullanıldığı için generator döner — bellekte tek seferde sadece bir parça tutulur.
            
def optimize_memory(df): #64-bitlik değişkenleri(float ve integer) 32-bit’e çevirerek %50’ye varan hafıza tasarrufu sağlar.
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
    numeric_data.fillna(0, inplace=True) #Sayısal verilerdeki NaN gibi eksik veya hatalı değerleri sıfırla değiştirmek.
    numeric_data.replace([np.inf, -np.inf], 0, inplace=True) #Sayısal verilerdeki inf, -inf gibi eksik veya hatalı değerleri sıfırla değiştirmek.

    print("NaN değer sayısı (sonrası):", numeric_data.isna().sum().sum())
    print("Sonsuz değer sayısı (sonrası):", np.isinf(numeric_data).sum().sum())
    return numeric_data

def preprocess_data_in_chunks(chunk):
    """
    Veriyi parçalara ayırarak işler.
    """
    if 'Label' not in chunk.columns: #Veri kümesinde hedef sütun (Label) yoksa hata fırlatılır.
        raise KeyError("Veri setinde 'Label' sütunu bulunamadı.")
    
    # Sayısal veriyi seçerken, sayısal olmayanları ayıklıyoruz
    numeric_data = chunk.select_dtypes(include=['float64', 'int64']) #numeric_data = chunk.select_dtypes(include=['float64', 'int64'])
    
    # Sayısal veri olup olmadığını kontrol et
    if numeric_data.empty:
        print("Uyarı: Sayısal veri bulunamadı. Lütfen veri setinizi kontrol edin.")
        return None
    
    # Bellek optimizasyonu
    numeric_data = optimize_memory(numeric_data)

    # NaN ve sonsuz değerleri temizleme
    numeric_data = clean_data(numeric_data) #numeric_data: Sadece sayısal sütunları içeren bir DataFrame.

    # NaN değerlerini doldurmak için SimpleImputer kullanıyoruz
    imputer = SimpleImputer(strategy='mean')  # İsterseniz 'median', 'most_frequent', 'constant' da kullanabilirsiniz, Eksik (NaN) verileri doldurmak için bir sınıf.
    
    try:
        # NaN'leri dolduruyoruz
        numeric_data_imputed = imputer.fit_transform(numeric_data) #fit_transform(...): Hem ortalamayı hesaplar (fit) hem de NaN’leri doldurur (transform).
    except ValueError as e:
        print(f"Error during imputation: {e}")
        return None  # Hata durumunda None döndürebiliriz

    # Dönüştürülen numpy.ndarray'i tekrar DataFrame'e dönüştürme
    numeric_data_imputed = pd.DataFrame(numeric_data_imputed, columns=numeric_data.columns)

    # Etiket sütununu geri ekle
    numeric_data_imputed['Label'] = chunk['Label']

    # Label sütununu kategorik hale getirme
    label_encoder = LabelEncoder() 
    numeric_data_imputed['Label'] = label_encoder.fit_transform(numeric_data_imputed['Label']) #Sınıflar 'Benign', 'Malware' gibi metin içeriyorsa, bunlar sayıya çevrilir (örn. 0, 1).

    # Sınıf dağılımını kontrol et ve SMOTE işlemi yap
    if numeric_data_imputed['Label'].nunique() > 1:  #Eğer veri setinde dengesizlik varsa (örneğin 95% Benign, 5% Attack gibi), SMOTE azınlık sınıfın kopyalarını oluşturarak veri setini dengeler.
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(
            numeric_data_imputed.drop('Label', axis=1), 
            numeric_data_imputed['Label']
        )
        # Hem X hem de y döndürüyoruz
        return X_resampled, y_resampled
    else:
        print("Sadece bir sınıf bulundu. SMOTE işlemi yapılamaz.") #SMOTE yalnızca 2 veya daha fazla sınıf varsa çalışır.
        return None
