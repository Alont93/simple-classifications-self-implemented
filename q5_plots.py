import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

MEAN_ZERO = 4
MEAN_ONE = 6
SAMPLES_FREQ_ZERO = 1/2
SAMPLES_FREQ_ONE = 1/2
STD = 1

def q5_one():
    x = np.linspace(stats.norm.ppf(0.01), 12.5, 200)

    pdfs0 = stats.norm.cdf(x, MEAN_ZERO, STD)
    pdfs1 = stats.norm.cdf(x, MEAN_ONE, STD)

    plt.plot(x, pdfs0, label="mean = " + str(MEAN_ZERO))
    plt.plot(x, pdfs1, label="mean = " + str(MEAN_ONE))
    plt.xlabel("samples")
    plt.ylabel("probability")
    plt.title("CDF of Gaussians")
    plt.legend()
    plt.show()

def q5_two():
    x = np.linspace(0, 10, 200)

    vhfunc = np.vectorize(h)
    hs = vhfunc(x)

    plt.plot(x, hs)
    plt.xlabel("samples")
    plt.ylabel("h(x)")
    plt.title("h(x) as a function of x")
    plt.show()


def h(x):
    w = STD * (MEAN_ONE - MEAN_ZERO)
    w_d_plus_one = -0.5 * MEAN_ONE * STD * MEAN_ONE + 0.5 * MEAN_ZERO * STD * MEAN_ZERO \
                   + np.log(SAMPLES_FREQ_ONE) - np.log(SAMPLES_FREQ_ZERO)

    w = np.array([w, w_d_plus_one])
    x = np.array([x, 1])

    e_exp = np.inner(x, w)
    return (np.e ** e_exp) / (np.e ** e_exp + 1)


def q5_three():
    x = np.random.normal(MEAN_ZERO, STD, 1000)
    x = np.sort(x)

    vhfunc = np.vectorize(h)
    hs = vhfunc(x)
    hs = np.cumsum(hs)

    plt.plot(x, hs)
    plt.xlabel("samples")
    plt.ylabel("h(x)")
    plt.title("h(x) as a function of x")
    plt.show()


q5_three()