import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

def grid_search_random_forest(X_train, y_train, X_test):
    print("X_train NaN değer sayısı:", np.isnan(X_train).sum().sum())
    print("X_train sonsuz değer sayısı:", np.isinf(X_train).sum().sum())
    
    print("y_train tipi:", type(y_train))
    
    # -----Neden sadece y_train?
    # Çünkü modelde etiketler (y_train) eksik (NaN) olamaz. Tahmin edilecek şey eksikse bu örneklerin hiçbir anlamı yoktur.
    # Aynı zamanda, X_train'de bu NaN etiketlere denk gelen satırları da silmemiz gerekir. Çünkü her girdinin bir etiketi olmalıdır.
    # -----Uygulama:
    # Sadece y_train'den NaN'leri sileriz ve sonra X_train'i bu silinmiş indekslerle uyumlu hale getiririz.
    if isinstance(y_train, pd.Series):
        print("y_train NaN değer sayısı:", y_train.isna().sum())
        y_train = y_train.dropna()  # NaN etiketleri (Label) varsa atılır (dropna()).
        X_train = X_train.loc[y_train.index] # Ardından X_train, sadece geçerli y_train indekslerine göre filtrelenir. Böylece X ile y uyumlu kalır.
    else:
        print("y_train NaN değer sayısı:", np.isnan(y_train).sum()) #Eğer y_train bir numpy dizisi ise aynı şekilde NaN sayısı kontrol edilir ama işlem yapılmaz (çünkü dizilerde indeks uyarlama yoktur).

    # NaN değerlerini doldurmak için SimpleImputer kullanıyoruz

    # ----Neden her ikisine de uygulanıyor?
    # Çünkü hem eğitim hem test verisinde eksik değerler (NaN) olabilir. Ancak:
    # fit_transform(X_train) ➝ Eğitim verisine göre ortalama hesaplanır.
    # transform(X_test) ➝ Test verisine bu hesaplanan ortalamalar uygulanır.
    # ----Neden test verisine fit yapılmaz?
    # Çünkü test verisini asla "görmemeliyiz", eğitim sırasında ona özel bir şey öğrenmemeliyiz.
    print("X_train'deki NaN değerler dolduruluyor...")
    imputer = SimpleImputer(strategy='mean')  # 'mean' yerine 'median' da kullanabilirsiniz
    X_train_imputed = imputer.fit_transform(X_train)  # NaN'leri X_train için dolduruyoruz, X_train verisine göre öğrenir ve uygular.
    X_test_imputed = imputer.transform(X_test)  # NaN'leri X_test için de dolduruyoruz, Aynı işlem X_test verisine de uygulanır, ancak fit edilmez (test verisi etkilenmez).
    
    # SMOTE ile veri dengeleme
    # ----Neden sadece eğitim verisine uygulanıyor?
    # SMOTE bir oversampling (çoğaltma) yöntemidir. Azınlık sınıfın örneklerini artırarak sınıflar arası dengeyi sağlar.
    # Test verisine SMOTE uygulanmaz. Çünkü bu, yapay örnekler ekleyerek testin doğallığını bozar ve modelin gerçek performansını çarpıtır.
    print("SMOTE ile veri dengeleme uygulanıyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

    # PCA uygulama (isteğe bağlı, daha fazla bileşen kullanılarak veri kaybı minimize edilebilir)
    # ----Neden her ikisine de uygulanıyor?
    # PCA, özellikleri (feature) azaltmak için kullanılır. Ancak fit sadece eğitim verisine yapılır.
    # fit_transform(X_train) ➝ PCA bileşenleri burada öğrenilir.
    # transform(X_test) ➝ Aynı dönüşüm, test verisine uygulanır.
    # Test verisine fit uygulanmaz çünkü bu overfitting'e yol açar.
    pca = PCA(n_components=0.98)  # Toplam varyansın %98’ini koruyacak kadar bileşen seçilir.
    X_train_resampled = pca.fit_transform(X_train_resampled) # Eğitim verisine uygulanır.
    X_test_pca = pca.transform(X_test_imputed)  # Test verisine aynı dönüşüm uygulanır.

    print(f"PCA sonrası yeni özellik sayısı: {X_train_resampled.shape[1]}")

    # GridSearchCV için parametre grid'i (daha dar bir grid kullanıyoruz)
    param_grid = {
        'n_estimators': [50, 100],  # Ağaç sayısını biraz daha daraltıyoruz
        'max_depth': [10, 20],  # Ağaç derinliğini daraltıyoruz
        'min_samples_split': [2, 5],  # İç düğümde minimum örnek sayısı
        'min_samples_leaf': [1, 2],  # Yaprak düğümde minimum örnek sayısı
        'class_weight': ['balanced']  # Sınıf dengesi
    }

    # Random Forest modelini başlatma
    rf_model = RandomForestClassifier(random_state=42) #Model başlatılır.

    # GridSearchCV başlatma
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1) #Grid içindeki tüm kombinasyonları 3 katlı çapraz doğrulama (cv=3) ile dener.

    # GridSearchCV ile model eğitimi
    # ----Neden sadece eğitim verisine uygulanıyor?
    # Bu adım modelin eğitildiği yerdir. Test verisi henüz değerlendirme için saklanır, bu aşamada kesinlikle kullanılmaz.
    grid_search.fit(X_train_resampled, y_train_resampled) #Model eğitilir ve en iyi hiperparametreler seçilir.

    # En iyi model parametreleri
    print("En iyi parametreler:", grid_search.best_params_)

    # En iyi model döndürülüyor
    return grid_search.best_estimator_, X_test_pca #GridSearch'ten gelen en iyi eğitimli model. Test verisi PCA'dan geçirilmiş haliyle değerlendirme için hazır.
