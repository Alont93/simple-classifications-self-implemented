import numpy as np
from numpy import random as rd
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


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


def check_Ni(test_set_lables, sorted_probabilities_indices, requested_TPR):
    needed_TP = test_set_lables.size * requested_TPR
    Ni = 0
    TP = 0

    while(TP < needed_TP):
        if(test_set_lables[sorted_probabilities_indices[Ni]] == 1):
            TP+=1
        Ni+=1

    return Ni



def q7():
    data = read_data()

    for i in range(1,10):
        test_iteration(data, i)

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend()
    plt.show()


def test_iteration(data, test_number):
    train_set, test_set = randomly_split_data(data)
    train_set_samples, train_set_labels = split_samples_and_lables(train_set)
    test_set_samples, test_set_labels = split_samples_and_lables(test_set)

    clf = LogisticRegression()
    clf.fit(train_set_samples, train_set_labels)

    predicted_probabilities = clf.predict_proba(test_set_samples)[:, 1]
    sorted_probabilities_indices = np.argsort(predicted_probabilities)[::-1]

    NP = np.count_nonzero(test_set_labels == 1)
    NN = test_set_labels.size - NP
    Nis = [0]

    for i in range(TEST_SET_AMOUNT):
        if test_set_labels[sorted_probabilities_indices[i]] == 1:
            Nis.append(i)

    TPR = np.arange(NP + 1) / NP
    Nis = np.array(Nis)
    FPR = (Nis - np.arange(NP + 1)) / NN

    TPR = np.hstack((TPR, 1))
    FPR = np.hstack((FPR, 1))

    plt.plot(FPR, TPR, label="test " + str(test_number))


q7()