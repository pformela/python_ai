import numpy as np
import matplotlib.pyplot as plt


def compute_model_output(x, w, b):
    # x - data, m exaples
    # w, b - model parameters
    # returns y - target values
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


# plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

print(f"x_train.shape: {x_train.shape}")

# m is the number of training examples
m = x_train.shape[0]
print(f"Number of training examples: {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# # plot the data points
# plt.scatter(x_train, y_train, marker='x', c='r')
# # title
# plt.title("Housing prices")
# # y-axis label
# plt.ylabel('Price (in 1000s of dollars)')
# # x-axis label
# plt.xlabel('Size (1000 sqft)')
# plt.show()

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

tmp_f_wb = compute_model_output(x_train, w, b,)

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} thousand dollars")


