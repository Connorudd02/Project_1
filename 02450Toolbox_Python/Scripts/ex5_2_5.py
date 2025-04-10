# exercise 5.2.5
import sklearn.linear_model as lm

# requires data from exercise 5.1.4
from ex5_1_5 import *
import matplotlib.pyplot as plt

# Split dataset into features and target vector
alcohol_idx = attributeNames.index("Alcohol")
y = X[:, alcohol_idx]

X_cols = list(range(0, alcohol_idx)) + list(range(alcohol_idx + 1, len(attributeNames)))
X = X[:, X_cols]

# Additional nonlinear attributes
fa_idx = attributeNames.index("Fixed acidity")
va_idx = attributeNames.index("Volatile acidity")
Xfa2 = np.power(X[:, fa_idx], 2).reshape(-1, 1)
Xva2 = np.power(X[:, va_idx], 2).reshape(-1, 1)
Xfava = (X[:, fa_idx] * X[:, va_idx]).reshape(-1, 1)
X = np.asarray(np.bmat("X, Xfa2, Xva2, Xfava"))

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X, y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est - y

# Display plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(y, y_est, ".g")
plt.xlabel("Alcohol content (true)")
plt.ylabel("Alcohol content (estimated)")

plt.subplot(4, 1, 3)
plt.hist(residual, 40)

plt.subplot(4, 3, 10)
plt.plot(Xfa2, residual, ".r")
plt.xlabel("Fixed Acidity ^2")
plt.ylabel("Residual")

plt.subplot(4, 3, 11)
plt.plot(Xva2, residual, ".r")
plt.xlabel("Volatile Acidity ^2")
plt.ylabel("Residual")

plt.subplot(4, 3, 12)
plt.plot(Xfava, residual, ".r")
plt.xlabel("Fixed*Volatile Acidity")
plt.ylabel("Residual")

plt.show()

print("Ran Exercise 5.2.5")
