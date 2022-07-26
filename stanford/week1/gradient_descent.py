import math, copy
import numpy as np
import matplotlib.pyplot as plt

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2

    total_cost = 1 / (2 * m) * cost_sum

    return total_cost

def compute_gradient(x, y, w, b):
    # x - data - m examples
    # y - target vals
    # w, b - scalar - model parameters
    # dj_dw - gradient of the cost w.r.t. the parameters w
    # dj_db - gradient of the cost w.r.t. the parameter b

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
        Performs gradient descent to fit w, b.
        Updates w, b by taking num_iters gradient steps with learning rate alpha

        x - data, m examples
        y - target examples
        w_in, b_in - initial vals of model parameters
        alpha - learning rate
        num-iters - number of iterations to run gradient descent
        cos_function - function to call to produce cost
        gradient_function - function to call to produce gradient

        Returns;
        w - scalar - updated value of w
        b - scalar - updated value of b
        J_history - history of cost vals
        p_history - history of params [w,b]
    """

    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update params using equation
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost of each iteration
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Print cost every 10 interval
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

