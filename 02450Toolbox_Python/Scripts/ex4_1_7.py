# exercise 4.1.7

import importlib_resources
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat

filename = importlib_resources.files("dtuimldmtools").joinpath("data/zipdata.mat")
# Digits to include in analysis (to include all, n = range(10) )
n = [1]

# Number of digits to generate from normal distributions
ngen = 10

# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat(filename)["traindata"]
X = traindata[:, 1:]
y = traindata[:, 0]
N, M = np.shape(X)  # or X.shape
C = len(n)

# Remove digits that are not to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = y == v
    class_mask = class_mask | cmsk
X = X[class_mask, :]
y = y[class_mask]
N = np.shape(X)[0]  # or X.shape[0]

mu = X.mean(axis=0)
s = X.std(ddof=1, axis=0)
S = np.cov(X, rowvar=0, ddof=1)

# Generate 10 samples from 1-D normal distribution
Xgen = np.random.randn(ngen, 256)
for i in range(ngen):
    Xgen[i] = np.multiply(Xgen[i], s) + mu

# Plot images
plt.figure()
for k in range(ngen):
    plt.subplot(2, int(np.ceil(ngen / 2.0)), k + 1)
    I = np.reshape(Xgen[k, :], (16, 16))
    plt.imshow(I, cmap=plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    if k == 1:
        plt.title("Digits: 1-D Normal")


# Generate 10 samples from multivariate normal distribution
Xmvgen = np.random.multivariate_normal(mu, S, ngen)
# Note if you are investigating a single class, then you may get:
# """RuntimeWarning: covariance is not positive-semidefinite."""
# Which in general is troublesome, but here is due to numerical imprecission


# Plot images
plt.figure()
for k in range(ngen):
    plt.subplot(2, int(np.ceil(ngen / 2.0)), k + 1)
    I = np.reshape(Xmvgen[k, :], (16, 16))
    plt.imshow(I, cmap=plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    if k == 1:
        plt.title("Digits: Multivariate Normal")

plt.show()

