# exercise 6.1.2

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

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc), K))
Error_test = np.empty((len(tc), K))

k = 0
for train_index, test_index in CV.split(X):
    print("Computing CV fold: {0}/{1}..".format(k + 1, K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion="gini", max_depth=t)
        dtc = dtc.fit(X_train, y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i, k], Error_train[i, k] = misclass_rate_test, misclass_rate_train
    k += 1


f = plt.figure()
plt.boxplot(Error_test.T)
plt.xlabel("Model complexity (max tree depth)")
plt.ylabel("Test error across CV folds, K={0})".format(K))

f = plt.figure()
plt.plot(tc, Error_train.mean(1))
plt.plot(tc, Error_test.mean(1))
plt.xlabel("Model complexity (max tree depth)")
plt.ylabel("Error (misclassification rate, CV K={0})".format(K))
plt.legend(["Error_train", "Error_test"])

plt.show()

print("Ran Exercise 6.1.2")
