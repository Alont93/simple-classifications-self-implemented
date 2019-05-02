import numpy as np
from numpy import random as rd
import pandas as pd

TEST_SET_AMOUNT = 1000
DATA_PATH = './spam.data.txt'

def randomly_split_data(data):
    data_size = data.shape[0]
    amount_of_training_data = int(data_size - TEST_SET_AMOUNT)
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


def q7():
    data = read_data()
    train_set, test_set = randomly_split_data(data)
    train_set_samples, train_set_labels = split_samples_and_lables(train_set)
    test_set_samples, test_set_samples = split_samples_and_lables(train_set)




q7()