import numpy as np


class knn:

    def __init__(self, k):
        self.neighbors_number = k
        self.features = None
        self.labels = None

    def fit(self, X, y):
        self.features = X
        self.labels = y

    # TODO: test this one
    def predict(self, x):
        dists_from_x = np.linalg.norm(self.features - x, axis=1)
        closest_points_indices = np.argpartition(dists_from_x, self.neighbors_number)[:self.neighbors_number]
        true_number_around = np.count_nonzero(self.features[closest_points_indices] == 1)

        if (self.neighbors_number - true_number_around) > self.neighbors_number / 2:
            return 1

        return 0
