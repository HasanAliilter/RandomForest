import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    # Model tahminleri
    y_pred = model.predict(X_test)

    # Veri türlerini kontrol et
    print(f"y_test türü: {type(y_test)}, y_pred türü: {type(y_pred)}")
    print(f"y_test örnek değerler: {y_test[:5]}")
    print(f"y_pred örnek değerler: {y_pred[:5]}")
    
    # NaN veya sonsuz değerleri kontrol et
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Etiketlerin hepsini string (metin) türüne dönüştür
    y_test = y_test.astype(str)
    y_pred = y_pred.astype(str)

    # Doğruluk ve sınıflandırma raporu hesaplama
    accuracy = (y_test == y_pred).mean()
    report = classification_report(y_test, y_pred)

    return accuracy, report