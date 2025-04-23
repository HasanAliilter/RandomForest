import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate              # Öğrenme oranı
        self.lambda_param = lambda_param     # Düzenleme katsayısı
        self.n_iters = n_iters              # İterasyon sayısı
        self.weights = None                 # Ağırlıklar (coef)
        self.bias = None                    # Bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Sınıfları (-1, 1) olacak şekilde dönüştür
        
        # Ağırlıkları ve bias'ı sıfırdan başlat
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Hinge Loss fonksiyonu kontrolü
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    # Marj içinde değilse, sadece düzenleme uygula
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    # Marj dışındaysa, hata yapıyorsa güncelle
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias
        return np.sign(approx)  # Tahmin sonucu -1 veya 1 olarak döner

# Örnek kullanım
if __name__ == "__main__":
    # Örnek eğitim verisi
    X_train = np.array([
        [2, 3], [10, 15], [5, 7], [1, 1], [7, 9], [9, 10],
        [8, 12], [3, 4], [6, 8], [11, 14]
    ])
    y_train = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

    # Model oluştur ve eğit
    model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    # Test verisi ile tahmin yap
    X_test = np.array([[6, 8], [3, 4], [10, 11]])
    predictions = model.predict(X_test)

    print(f"Predictions: {predictions}")
