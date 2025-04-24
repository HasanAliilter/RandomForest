import numpy as np

class MySVM:
    def __init__(self, ogrenme_orani=0.01, lambda_orani=0.01, tekrar=1000):
        self.oran = ogrenme_orani
        self.lambda_ = lambda_orani
        self.tekrar = tekrar
        self.w = None
        self.b = None

    def egit(self, X, y):
        satir_sayisi, ozellik_sayisi = X.shape
        y_etiket = np.where(y <= 0, -1, 1)

        self.w = np.zeros(ozellik_sayisi)
        self.b = 0

        for i in range(self.tekrar):
            for j, x in enumerate(X):
                kontrol = y_etiket[j] * (np.dot(x, self.w) + self.b) >= 1

                if kontrol:
                    self.w -= self.oran * (2 * self.lambda_ * self.w)
                else:
                    self.w -= self.oran * (2 * self.lambda_ * self.w - np.dot(x, y_etiket[j]))
                    self.b -= self.oran * y_etiket[j]

    def tahmin_et(self, X):
        sonuc = np.dot(X, self.w) + self.b
        return np.sign(sonuc)

if __name__ == "__main__":
    X_train = np.array([
        [2, 3], [10, 15], [5, 7], [1, 1], [7, 9], [9, 10],
        [8, 12], [3, 4], [6, 8], [11, 14]
    ])
    y_train = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

    svm = MySVM(ogrenme_orani=0.001, lambda_orani=0.01, tekrar=1000)
    svm.egit(X_train, y_train)

    X_test = np.array([[6, 8], [3, 4], [10, 11]])
    tahminler = svm.tahmin_et(X_test)

    print("Tahminler:", tahminler)
