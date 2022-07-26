import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl


def compute_cost(x, y, w, b):
    # x - data, m examples
    # y -target vals
    # w, b - scalar - model parameters
    # returns total_cost of using w, b parameters for linear regression

    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

