# exercise 5.2.3

import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# Parameters
Kd = 5  # no of terms for data generator
Km = 3  # no of terms for regression model
N = 50  # no of data objects to train a model
Xe = np.linspace(-2, 2, 1000).reshape(
    -1, 1
)  # X values to visualize true data and model
eps_mean, eps_std = 0, 0.5  # noise parameters

# Generate dataset (with noise)
X = np.linspace(-2, 2, N).reshape(-1, 1)
Xd = np.power(X, range(1, Kd + 1))
eps = eps_std * np.random.randn(N) + eps_mean
w = -np.power(-0.9, range(1, Kd + 2))
y = w[0] + Xd @ w[1:] + eps


# True data generator (assuming no noise)
Xde = np.power(Xe, range(1, Kd + 1))
y_true = w[0] + Xde @ w[1:]


# Fit ordinary least squares regression model
Xm = np.power(X, range(1, Km + 1))
model = lm.LinearRegression()
model = model.fit(Xm, y)

# Predict values
Xme = np.power(Xe, range(1, Km + 1))
y_est = model.predict(Xme)

# Plot original data and the model output
f = plt.figure()
plt.plot(X, y, ".")
plt.plot(Xe, y_true, "-")
plt.plot(Xe, y_est, "-")
plt.xlabel("X")
plt.ylabel("y")
plt.ylim(-2, 8)
plt.legend(
    [
        "Training data",
        "Data generator K={0}".format(Kd),
        "Regression fit (model) K={0}".format(Km),
    ]
)

plt.show()

print("Ran Exercise 5.2.3")
