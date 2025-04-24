import numpy as np

# Decision Tree Classifier (manuel)
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / m
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None

        m, n = X.shape

        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold

                if np.sum(left_indices) < self.min_samples_split or np.sum(right_indices) < self.min_samples_split:
                    continue

                left_gini = self._gini(y[left_indices])
                right_gini = self._gini(y[right_indices])
                weighted_gini = (np.sum(left_indices) * left_gini + np.sum(right_indices) * right_gini) / m

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        return best_split

    def _build_tree(self, X, y, depth):
        # Stopping Conditions (Durdurma Koşulları)
        if len(np.unique(y)) == 1:  # Aynı sınıfa aitse
            return {'value': y[0]}
        if self.max_depth and depth >= self.max_depth:
            return {'value': np.bincount(y).argmax()}
        if len(y) < self.min_samples_split:
            return {'value': np.bincount(y).argmax()}

        # En iyi bölmeyi bul
        split = self._best_split(X, y)
        if not split:
            return {'value': np.bincount(y).argmax()}

        left_tree = self._build_tree(X[split['left_indices']], y[split['left_indices']], depth + 1)
        right_tree = self._build_tree(X[split['right_indices']], y[split['right_indices']], depth + 1)

        return {
            'feature_idx': split['feature_idx'],
            'threshold': split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _predict_sample(self, x, tree):
        if 'value' in tree:
            return tree['value']

        feature_idx = tree['feature_idx']
        threshold = tree['threshold']

        if x[feature_idx] <= threshold:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


# RandomForest Classifier (manuel)
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        num_samples, num_features = X.shape

        if not self.max_features:
            self.max_features = int(np.sqrt(num_features))

        self.trees = []
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(num_samples, num_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            feature_indices = np.random.choice(num_features, self.max_features, replace=False)
            X_sample = X_sample[:, feature_indices]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)

            self.trees.append((tree, feature_indices))

    def predict(self, X):
        tree_predictions = np.array([
            tree.predict(X[:, feature_indices]) for tree, feature_indices in self.trees
        ])

        majority_votes = np.array([
            np.bincount(tree_predictions[:, i]).argmax() for i in range(X.shape[0])
        ])

        return majority_votes


# Test Etme
if __name__ == "__main__":
    X_train = np.array([
        [2, 3], [10, 15], [5, 7], [1, 1], [7, 9], [9, 10],
        [8, 12], [3, 4], [6, 8], [11, 14]
    ])
    y_train = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

    forest = RandomForest(n_estimators=5, max_depth=3)
    forest.fit(X_train, y_train)

    X_test = np.array([[6, 8], [3, 4], [10, 11]])
    predictions = forest.predict(X_test)

    print(f"Predictions: {predictions}")
