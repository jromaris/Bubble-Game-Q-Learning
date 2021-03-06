from scipy import integrate
from numpy import cosh, exp
import numpy as np
import matplotlib.pyplot as plt


def genlog_func(t, epsilon_params):
    t = 5*t - 1.5
    a = epsilon_params['a']
    k = epsilon_params['k']
    b = epsilon_params['b']
    q = epsilon_params['q']
    v = epsilon_params['v']
    m = epsilon_params['m']
    c = epsilon_params['c']
    # a, k, b, q, v, m, c = 0, 1, 1.5, 0.5, 0.12, 0, 1
    if epsilon_params['constant'][0]:
        y = epsilon_params['constant'][1]
    else:
        y = a + (k - a) / (c + q * exp(-b*t))**(1/v)
    return y


def genlogistic(epsilon_params):

    xs = [x/1000 for x in range(0, 1001)]
    results = np.array([genlog_func(x, epsilon_params) for x in xs])
    plt.plot(results)
    plt.show()
