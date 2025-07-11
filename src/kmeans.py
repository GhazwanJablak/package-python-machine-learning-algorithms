import numpy as np
from src.utils import euclidean_distance

class KmeansClustering():
    
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations
    
    def predict(self, X):
        self.n_samples, self.n_features = X.shape
        for _ in range(self.iterations):

            centroids_idx = np.random.choice(X.shape[0], 4, replace=False)
            self.centroids = [X[idx] for idx in centroids_idx]

            self.clusters = self.create_clusters(X)
            old_centroids = self.centroids
            self.centroids = self.new_centroids(X)
            if self.is_converge(old_centroids):
                break
        labels = self.get_labels()
        return labels
        
    
    @staticmethod
    def get_centroids(point, centroids):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        idx = np.argmin(distances)
        return idx
    
    def create_clusters(self, X):
        self.clusters = [[] for _ in range(self.k)]
        for idx, x in enumerate(X):
            centroid = self.get_centroids(x, self.centroids)
            self.clusters[centroid].append(idx)
        return self.clusters

    def new_centroids(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        for idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
        centroids[idx] = cluster_mean
        return centroids
    
    def is_converge(self, new_centroids):
        val = sum([euclidean_distance(new_centroids[i], self.centroids[i]) for i in range(self.k)])
        return val==0

    def get_labels(self):
        labels = np.empty(self.n_samples)
        for idx, item in enumerate(self.clusters):
            for val in item:
                labels[val]=idx
        return labels