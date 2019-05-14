import numpy as np
import scipy.stats as stats
from scipy.special import logit
import matplotlib.pyplot as plt


MEAN_ZERO = 4
MEAN_ONE = 6
SAMPLES_FREQ_ZERO = 1/2
SAMPLES_FREQ_ONE = 1/2
STD = 1

T_VALUES = [0.2, 0.4, 0.55, 0.95]


def q5_one():
    normal_cdf()
    normal_pdf()


def normal_cdf():
    x = np.linspace(stats.norm.ppf(0.01), 12.5, 200)
    cdfs0 = stats.norm.cdf(x, MEAN_ZERO, STD)
    cdfs1 = stats.norm.cdf(x, MEAN_ONE, STD)
    plt.plot(x, cdfs0, label="mean = " + str(MEAN_ZERO))
    plt.plot(x, cdfs1, label="mean = " + str(MEAN_ONE))
    plt.xlabel("samples")
    plt.ylabel("probability")
    plt.title("CDF of Gaussians")
    plt.legend()
    plt.show()


def normal_pdf():
    x = np.linspace(stats.norm.ppf(0.01), 12.5, 200)
    pdfs0 = stats.norm.cdf(x, MEAN_ZERO, STD)
    pdfs1 = stats.norm.cdf(x, MEAN_ONE, STD)
    plt.plot(x, pdfs0, label="mean = " + str(MEAN_ZERO))
    plt.plot(x, pdfs1, label="mean = " + str(MEAN_ONE))
    plt.xlabel("samples")
    plt.ylabel("probability")
    plt.title("PDF of Gaussians")
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
    cdf_of_hx_for_0()
    cdf_of_hx_for_1()


def cdf_of_hx_for_0():
    x = np.linspace(0, 1, 200)
    hx = stats.norm.cdf((1 / 2) * logit(x) + 5, MEAN_ZERO, STD)
    plt.plot(x, hx)
    plt.title("CDF of h(x) for x~X|(Y=0)")
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.show()


def cdf_of_hx_for_1():
    x = np.linspace(0, 1, 200)
    hx = stats.norm.cdf((1 / 2) * logit(x) + 5, MEAN_ONE, STD)
    plt.plot(x, hx)
    plt.title("CDF of h(x) for x~X|(Y=1)")
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.show()


def q5_four():
    x = np.linspace(0, 1, 200)
    hz_0 = stats.norm.cdf((1 / 2) * logit(x) + 5, MEAN_ZERO, STD)
    hz_1 = stats.norm.cdf((1 / 2) * logit(x) + 5, MEAN_ONE, STD)
    plt.plot(x, 1 - hz_0, label='CDF of h(Z1)')
    plt.plot(x, 1 - hz_1, label='CDF of h(Z2)')
    plt.title("1 minus CDF of h(Z1) and h(Z2)")
    plt.xlabel("h(Z)")
    plt.ylabel("probability")
    plt.show()


def q5_five():
    for t in T_VALUES:
        fpr = 1 - stats.norm.cdf((1 / 2) * logit(t) + 5, MEAN_ZERO, STD)
        tpr = 1 - stats.norm.cdf((1 / 2) * logit(t) + 5, MEAN_ONE, STD)
        print("for t=" + str(t) + ": FPR(t)=" + str(fpr) + ", TPR(t)=" + str(tpr))


def q5_six():
    x = np.linspace(0, 10, 200)
    x_points = []
    for i in range(len(T_VALUES)):
        t = T_VALUES[i]
        x_points.append(0.5 * logit(t) + 5)

    plt.plot(x, stats.norm.pdf(x, MEAN_ZERO, STD), label='mean = ' + str(MEAN_ZERO))
    plt.plot(x, stats.norm.pdf(x, MEAN_ONE, STD), label='mean = ' + str(MEAN_ONE))

    colors = ['r', 'k', 'y', 'c']
    for i in range(len(T_VALUES)):
        t = T_VALUES[i]
        plt.axvline(x_points[i], linestyle='--', c=colors[i], label='t = '+str(t))

    plt.title("Threshold Minimal Value")
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.legend()
    plt.show()


def q5_seven():
    x = np.linspace(0, 1, 200)
    fpr = 1 - stats.norm.cdf((1 / 2) * logit(x) + 5, MEAN_ZERO, STD)
    tpr = 1 - stats.norm.cdf((1 / 2) * logit(x) + 5, MEAN_ONE, STD)

    plt.plot(fpr, tpr)
    plt.title("ROC Curve of h")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()
