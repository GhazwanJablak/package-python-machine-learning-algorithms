import numpy as np
from src.utils import euclidean_distance

class KNNClassifier():
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_point(self, x):
        distances = [euclidean_distance(x, x2) for x2 in self.X]
        indexes = np.argsort(distances)[:self.k]
        closest_points = [self.y[i] for i in indexes]
        point_class = np.median(closest_points)
        return point_class
    
    def predict(self, X_test):
        predictions = [self.predict_point(x) for x in X_test]
        return predictions


class KNNRegressor():
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_point(self, x):
        distances = [euclidean_distance(x, x2) for x2 in self.X_train]
        idx = np.argsort(distances)[:self.k]
        point_prediction = np.mean(self.y_train[idx])
        return point_prediction
    
    def predict(self, X_test):
        predictions = [self.predict_point(x) for x in X_test]
        return predictions
