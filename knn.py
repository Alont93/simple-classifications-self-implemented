import numpy as np


class knn:

    def __init__(self, k):
        self.__k = k
        self.__features = None
        self.__labels = None

    def fit(self, X, y):
        self.__features = X
        self.__labels = y

    def predict(self, x):
        dist_from_x = np.linalg.norm(self.__features - x, axis=1)
        closet_indices = np.argpartition(dist_from_x, self.__k)[ : self.__k]
        true_neighbours = np.sum(self.__labels[closet_indices])

        if true_neighbours > self.__k / 2:
            return 1

        return 0