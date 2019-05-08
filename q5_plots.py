import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

MEAN_ZERO = 4
MEAN_ONE = 6
SAMPLES_FREQ_ZERO = 1/2
SAMPLES_FREQ_ONE = 1/2
STD = 1

def q5_one():

    x = np.linspace(stats.norm.ppf(0.01), 10, 100)
    rand = []

    pdfs = stats.norm.pdf(x, MEAN_ZERO, STD)

    plt.plot(x, pdfs)
    plt.show()


q5_one()