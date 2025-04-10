# exercise 10.1.1
import importlib_resources
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

from dtuimldmtools import clusterplot

filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth1.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
attributeNames = [name[0] for name in mat_data["attributeNames"].squeeze()]
classNames = [name[0][0] for name in mat_data["classNames"]]
N, M = X.shape
C = len(classNames)

# Number of clusters:
K = 4

# K-means clustering:
centroids, cls, inertia = k_means(X, K)

# Plot results:
plt.figure(figsize=(14, 9))
clusterplot(X, cls, centroids, y)
plt.show()

print("Ran Exercise 10.1.1")
