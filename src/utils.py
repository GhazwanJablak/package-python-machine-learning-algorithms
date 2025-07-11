import numpy as np

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1-x2)

def accuracy(y, y_pred):
    accuracy = np.sum(y == y_pred) / len(y)
    return accuracy
