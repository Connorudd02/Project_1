# exercise 5.2.4
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# requires wine data from exercise 5.1.5
from ex5_1_5 import *

# Split dataset into features and target vector
alcohol_idx = attributeNames.index("Alcohol")
y = X[:, alcohol_idx]

X_cols = list(range(0, alcohol_idx)) + list(range(alcohol_idx + 1, len(attributeNames)))
X = X[:, X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X, y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est - y

# Display scatter plot
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(y, y_est, ".")
plt.xlabel("Alcohol content (true)")
plt.ylabel("Alcohol content (estimated)")
plt.subplot(2, 1, 2)
plt.hist(residual, 40)

plt.show()

print("Ran Exercise 5.2.4")
