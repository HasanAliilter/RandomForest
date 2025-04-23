import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            return {'value': np.mean(y)}
        
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return {'value': np.mean(y)}
        
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _best_split(self, X, y):
        best_feature, best_threshold, best_loss = None, None, float('inf')
        num_samples, num_features = X.shape
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                loss = self._calculate_loss(y[left_indices], y[right_indices])
                
                if loss < best_loss:
                    best_feature = feature
                    best_threshold = threshold
                    best_loss = loss
        
        return best_feature, best_threshold
    
    def _calculate_loss(self, left_y, right_y):
        left_loss = np.var(left_y) * len(left_y)
        right_loss = np.var(right_y) * len(right_y)
        return left_loss + right_loss
    
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])
    
    def _predict(self, inputs):
        node = self.tree
        while 'feature' in node:
            if inputs[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']

class XGBoost:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        y_pred = np.zeros(len(y))  # Başlangıç tahminlerini sıfırla

        for _ in range(self.n_estimators):
            # Gradients (Log-loss türevi)
            gradient = y - self._sigmoid(y_pred)
            
            # Karar ağacı ile gradient'i öğren
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, gradient)
            self.trees.append(tree)

            # Yeni ağacın çıktısını learning rate ile güncelle
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return (self._sigmoid(y_pred) > 0.5).astype(int)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Örnek Kullanım
if __name__ == "__main__":
    # Örnek eğitim verisi
    X_train = np.array([
        [2, 3], [10, 15], [5, 7], [1, 1], [7, 9], [9, 10],
        [8, 12], [3, 4], [6, 8], [11, 14]
    ])
    y_train = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

    # Model oluştur ve eğit
    model = XGBoost(n_estimators=5, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Test verisi ile tahmin yap
    X_test = np.array([[6, 8], [3, 4], [10, 11]])
    predictions = model.predict(X_test)

    print(f"Predictions: {predictions}")
