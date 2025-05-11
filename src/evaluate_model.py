import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# model: Eğitilmiş (trained) makine öğrenimi modeli.
# X_test: Test veri setindeki girdiler (özellikler).
# y_test: Test veri setindeki gerçek etiketler (doğru sınıflar).
def evaluate_model(model, X_test, y_test):
    # Model tahminleri
    # model.predict(X_test) ifadesi, modelin test verisi üzerindeki tahminlerini üretir.
    # Sonuç: y_pred, modelin X_test için ürettiği tahmin etiketleridir.
    y_pred = model.predict(X_test)

    # Veri türlerini kontrol et
    # type(...): Hem y_test hem y_pred'in veri türünü ekrana yazdırır. Genelde bunlar numpy.ndarray veya pandas.Series olabilir.
    # [:5]: İlk 5 elemanı yazdırır. Bu, çıktının mantıklı olup olmadığını hızlıca kontrol etmenizi sağlar.
    print(f"y_test türü: {type(y_test)}, y_pred türü: {type(y_pred)}")
    print(f"y_test örnek değerler: {y_test[:5]}")
    print(f"y_pred örnek değerler: {y_pred[:5]}")
    
    # NaN veya sonsuz değerleri kontrol et
    # Eğer y_test veya y_pred bir pandas.Series ise .values özelliği ile onları NumPy dizisine (ndarray) dönüştürüyoruz.
    # Neden? Bazı fonksiyonlar NumPy array ile daha stabil çalışır, karşılaştırmalar da daha net olur.
    if isinstance(y_test, pd.Series):  # Pandas Series ise
        y_test = y_test.values
    if isinstance(y_pred, pd.Series):  # Pandas Series ise
        y_pred = y_pred.values

    # Etiketlerin hepsini string (metin) türüne dönüştür
    # Her iki etiket dizisi de str (string) türüne çevrilir.
    # Neden gerekli?
    # Bazı durumlarda etiketler hem sayı hem string olabilir (örneğin 0, '0', 1.0, '1' gibi).
    # Karşılaştırmalarda bu tür farklar hatalara yol açabilir. Hepsi str yapılarak tutarlılık sağlanır.
    y_test = y_test.astype(str)
    y_pred = y_pred.astype(str)

    # Doğruluk ve sınıflandırma raporu hesaplama
    accuracy = (y_test == y_pred).mean()
    report = classification_report(y_test, y_pred)

    return accuracy, report