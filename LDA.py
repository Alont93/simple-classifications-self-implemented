import numpy as np
from functools import reduce

RELEVANT_FEATURES = [1, 4, 21, 32, 22]

class LDA:

    def __init__(self):
        self.__frequencies = None
        self.__means = None
        self.__covariance_matrix = None
        self.__num_of_labels = None
        self.__labels = None
        self.__number_of_features = None

    def fit(self, X, y):

        X = X[:, RELEVANT_FEATURES]

        number_of_samples = y.size
        number_of_features = X.shape[1]
        self.__number_of_features = number_of_features

        unique, counts = np.unique(y, return_counts=True)
        self.__labels = unique
        self.__frequencies = counts / number_of_samples

        self.__num_of_labels = unique.size
        self.__means = np.zeros((self.__num_of_labels, number_of_features))

        sorted_indices = np.argsort(y)
        X_sorted_by_labels = X[sorted_indices]
        X_splited_by_labels = np.array_split(X_sorted_by_labels, np.where(np.diff(y[sorted_indices]) != 0)[0] + 1)

        for i in range(self.__num_of_labels):
            self.__means[i] = np.sum(X_splited_by_labels[i], axis=0) / counts[i]

        partial_covariances = []
        for i in range(self.__num_of_labels):
            X_minus_mean = X_splited_by_labels[i] - self.__means[i]
            partial_covariances.append(X_minus_mean.T @ X_minus_mean)

        self.__covariance_matrix = reduce(np.add, partial_covariances) / (number_of_samples - 2)

    def predict(self, x):

        x = x[RELEVANT_FEATURES]

        sigma_inv = np.linalg.inv(self.__covariance_matrix)
        xT = x.reshape((1, self.__number_of_features))

        probabilities = np.zeros_like(self.__labels)
        for i in range(self.__num_of_labels):
            label_mean = self.__means[i].reshape(self.__number_of_features, 1)
            probabilities[i] = xT @ sigma_inv @ label_mean \
                               - 0.5 * label_mean.T @ sigma_inv @ label_mean + np.log(self.__frequencies[i])

        max_probability_index = np.argmax(probabilities)
        return self.__labels[max_probability_index]