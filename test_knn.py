import numpy as np
from numpy import random as rd
import pandas as pd
import matplotlib.pyplot as plt

import knn

Ks = [1, 2, 5, 10, 100]
TEST_SET_AMOUNT = 1000
DATA_PATH = './spam.data.txt'

def randomly_split_data(data):
    data_size = data.shape[0]
    amount_of_training_data = data_size - TEST_SET_AMOUNT
    training_set_indices = rd.choice(data_size, size=amount_of_training_data, replace=False)

    training_data = data[training_set_indices, :]
    test_data = np.delete(data, training_set_indices, axis=0)

    return training_data, test_data


def read_data():
    data_table = pd.read_csv(DATA_PATH, sep=' ', header=None)
    data_mat = data_table.values
    return data_mat.astype(np.float64)


def split_samples_and_lables(data):
    return data[:,0:-1], data[:,-1]

def q7c():
    data = read_data()

    for i in range(1,11):
        test_iteration(data, i)

    plt.xlabel("K")
    plt.ylabel("Error")
    plt.title("Error as a function of K")
    plt.legend()
    plt.show()


def test_iteration(data, test_number):
    train_set, test_set = randomly_split_data(data)
    train_set_samples, train_set_labels = split_samples_and_lables(train_set)
    test_set_samples, test_set_labels = split_samples_and_lables(test_set)

    ks_array = np.array(Ks)
    errors = np.zeros_like(ks_array).astype(np.float64)

    i = 0
    for k in Ks:
        knn_learner = knn.knn(k)
        knn_learner.fit(train_set_samples, train_set_labels)
        predictions = np.apply_along_axis(knn_learner.predict, 1, test_set_samples)

        error = np.sum(np.square(predictions - test_set_labels)) / TEST_SET_AMOUNT
        errors[i] = error
        i += 1

    plt.plot(ks_array, errors, label="test " + str(test_number), marker='o')



q7c()