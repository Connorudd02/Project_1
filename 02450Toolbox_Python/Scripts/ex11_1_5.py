# exercise 11.1.5
import importlib_resources
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.mixture import GaussianMixture

filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth2.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
attributeNames = [name[0] for name in mat_data["attributeNames"].squeeze()]
classNames = [name[0][0] for name in mat_data["classNames"]]
N, M = X.shape
C = len(classNames)


# Range of K's to try
KRange = range(1, 11)
T = len(KRange)

covar_type = "full"  # you can try out 'diag' as well
reps = 3  # number of fits with different initalizations, best result will be kept
init_procedure = "kmeans"  # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10, shuffle=True)

for t, K in enumerate(KRange):
    print("Fitting model for K={0}".format(K))

    # Fit Gaussian mixture model
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=covar_type,
        n_init=reps,
        init_params=init_procedure,
        tol=1e-6,
        reg_covar=1e-6,
    ).fit(X)

    # Get BIC and AIC
    BIC[t,] = gmm.bic(X)
    AIC[t,] = gmm.aic(X)

    # For each crossvalidation fold
    for train_index, test_index in CV.split(X):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        X_test = X[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GaussianMixture(
            n_components=K, covariance_type=covar_type, n_init=reps
        ).fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score_samples(X_test).sum()


# Plot results

plt.figure(1)
plt.plot(KRange, BIC, "-*b")
plt.plot(KRange, AIC, "-xr")
plt.plot(KRange, 2 * CVE, "-ok")
plt.legend(["BIC", "AIC", "Crossvalidation"])
plt.xlabel("K")
plt.show()

print("Ran Exercise 11.1.5")
