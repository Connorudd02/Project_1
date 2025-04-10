# exercise 6.1.1

import importlib_resources
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection, tree

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine2.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
attributeNames = [name[0] for name in mat_data["attributeNames"][0]]
classNames = [name[0][0] for name in mat_data["classNames"]]
N, M = X.shape
C = len(classNames)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=test_proportion
)

# Initialize variables
Error_train = np.empty((len(tc), 1))
Error_test = np.empty((len(tc), 1))

for i, t in enumerate(tc):
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion="gini", max_depth=t)
    dtc = dtc.fit(X_train, y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = np.asarray(dtc.predict(X_test), dtype=int)
    y_est_train = np.asarray(dtc.predict(X_train), dtype=int)
    misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
    misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
    Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train

f = plt.figure()
plt.plot(tc, Error_train * 100)
plt.plot(tc, Error_test * 100)
plt.xlabel("Model complexity (max tree depth)")
plt.ylabel("Error (%)")
plt.legend(["Error_train", "Error_test"])

plt.show()

print("Ran Exercise 6.1.1")
