import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Load dataset
seeds_dataset = "Data/seeds_dataset.txt"
data = np.loadtxt(seeds_dataset)

# Assume the features are in all columns except the last, and that the last column is the target
X = data[:, :-1]
y = data[:, -1]

# We choose 50 values logarithmically spaced between 10^-4 and 10^4.
lambdas = np.logspace(-4, 4, 50)
mse_errors = []

# Set up 10-fold cross-validation.
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for l in lambdas:
    model = Ridge(alpha=l)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse = -scores.mean()  # Convert negative MSE to positive MSE.
    mse_errors.append(mse)

errors = np.array(mse_errors)
mse_errors = np.array(mse_errors)
rmse_errors = np.sqrt(mse_errors)
y_mean = np.mean(y)
error_percentages = (rmse_errors / y_mean) * 100

# Find the index of the minimum error
min_error_index = np.argmin(errors)
optimal_lambda = lambdas[min_error_index]
min_error_percentage = errors[min_error_index]

print("Minimum generalization error (percentage RMSE): {:.2f}%".format(min_error_percentage))
print("Optimal lambda: {:.4f}".format(optimal_lambda))

# Fit the Ridge regression model on the entire dataset.
model_opt = Ridge(alpha=optimal_lambda)
model_opt.fit(X, y)

# Extract intercept and coefficients.
intercept = model_opt.intercept_
coeffs = model_opt.coef_

# Create feature names if none are provided.
feature_names = [f"x{i+1}" for i in range(X.shape[1])]

# Build the equation string.
equation = f"y = {intercept:.4f}"
for coef, fname in zip(coeffs, feature_names):
    # Include a '+' sign if coefficient is positive; otherwise, use '-'
    sign = " + " if coef >= 0 else " - "
    equation += f"{sign}{abs(coef):.4f}*{fname}"

print("\nLinear Model Equation:")
print(equation)

# Plotting generalization error vs lambda.
plt.figure(figsize=(8, 6))
plt.semilogx(lambdas, errors, marker='o', linestyle='-')
plt.xlabel('Regularization Parameter λ')
plt.ylabel('Estimated Generalization Error (MSE)')
plt.title('Generalization Error as a Function of λ')
plt.grid(True)
plt.show()





