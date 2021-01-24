from scipy import integrate
from numpy import cosh, exp
import numpy as np
import matplotlib.pyplot as plt


def fuchi(x):
    return 1 / cosh(x)


def gudermannian():
    xs = [x/200 for x in range(-1000, 1001)]
    results = np.array([integrate.quad(fuchi, 0, x) for x in xs])
    plt.plot(results)
    plt.show()


def genlog_func(t):
    t = 5*t - 1.5
    a, k, b, q, v, m, c = 0, 1, 1.5, 0.5, 0.2, 0, 1
    y = a + (k - a) / (c + q * exp(-b*t))**(1/v)
    return y


def genlogistic():

    xs = [x/1000 for x in range(0, 1001)]
    results = np.array([genlog_func(x) for x in xs])
    plt.plot(results)
    plt.show()
