# exercise 5.2.2

import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# Use dataset as in the previous exercise
N = 100
X = np.array(range(N)).reshape(-1, 1)
eps_mean, eps_std = 0, 0.1
eps = np.array(eps_std * np.random.randn(N) + eps_mean).reshape(-1, 1)
w0 = -0.5
w1 = 0.01
y = w0 + w1 * X + eps
y_true = y - eps

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X, y)
# Compute model output:
y_est = model.predict(X)
# Or equivalently:
# y_est = model.intercept_ + X @ model.coef_


# Plot original data and the model output
f = plt.figure()

plt.plot(X, y, ".")
plt.plot(X, y_true, "-")
plt.plot(X, y_est, "-")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(["Training data", "Data generator", "Regression fit (model)"])

plt.show()

print("Ran Exercise 5.2.2")
