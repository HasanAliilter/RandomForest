import numpy as np

class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(model)
            error = y_pred - y

            # Ağırlıkları ve bias'ı güncelle
            self.weights -= self.learning_rate * np.dot(X.T, error) / m
            self.bias -= self.learning_rate * np.sum(error) / m

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(model)
        return (y_pred >= 0.5).astype(int)

# Basit deneme
if __name__ == "__main__":
    X_train = np.array([
        [1, 2],
        [2, 1],
        [2, 3],
        [3, 5],
        [5, 2],
        [6, 1]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = SimpleLogisticRegression()
    model.fit(X_train, y_train)

    X_test = np.array([[1, 3], [4, 2]])
    predictions = model.predict(X_test)
    print("Tahminler:", predictions)
