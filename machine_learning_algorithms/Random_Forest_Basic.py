import numpy as np
from collections import Counter

#ağırlık ve veri türü dedim
X = np.array([
    [150, 1],
    [130, 1],
    [180, 0],
    [80,  1],
    [70,  1],
    [60,  0],
])
y = np.array([0, 0, 0, 1, 1, 1])

sample = np.array([140, 1])

def tree_1(x):
    return 0 if x[0] > 90 else 1

def tree_2(x):
    return 1 if x[1] == 1 else 0

def tree_3(x):
    return 0 if x[0] > 110 else 1

trees = [tree_1, tree_2, tree_3]
predictions = [tree(sample) for tree in trees]

final_prediction = Counter(predictions).most_common(1)[0][0]

print("Tahminler:", predictions)
print("Sonuç (0: a, 1: b):", final_prediction)
