# exercise 10.2.1
import importlib_resources
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.io import loadmat

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


# Perform hierarchical/agglomerative clustering on data matrix
Method = "single"
Metric = "euclidean"

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 4
cls = fcluster(Z, criterion="maxclust", t=Maxclust)
plt.figure(1)
clusterplot(X, cls.reshape(cls.shape[0], 1), y=y)

# Display dendrogram
max_display_levels = 6
plt.figure(2, figsize=(10, 4))
dendrogram(
    Z, truncate_mode="level", p=max_display_levels, color_threshold=Z[-Maxclust + 1, 2]
)

plt.show()

print("Ran Exercise 10.2.1")
