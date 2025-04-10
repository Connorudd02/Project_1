# exercise 11.2.3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Draw samples from mixture of gaussians (as in exercise 11.1.1)
N = 1000
M = 1
x = np.linspace(-10, 10, 50)
X = np.empty((N, M))
m = np.array([1, 3, 6])
s = np.array([1, 0.5, 2])
c_sizes = np.random.multinomial(N, [1.0 / 3, 1.0 / 3, 1.0 / 3])
for c_id, c_size in enumerate(c_sizes):
    X[
        c_sizes.cumsum()[c_id] - c_sizes[c_id] : c_sizes.cumsum()[c_id], :
    ] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size, M))


# Number of neighbors
K = 200

# x-values to evaluate the KNN
xe = np.linspace(-10, 10, 100)

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(
    np.expand_dims(xe, axis=-1)
)  # note expand_dims is simple to make it (100,1) and not (100,) array

# Compute the density
knn_density = 1.0 / (D[:, 1:].sum(axis=1) / K)

# Compute the average relative density
DX, iX = knn.kneighbors(X)
knn_densityX = 1.0 / (DX[:, 1:].sum(axis=1) / K)
knn_avg_rel_density = knn_density / (knn_densityX[i[:, 1:]].sum(axis=1) / K)


# Plot KNN density
plt.figure(figsize=(6, 7))
plt.subplot(2, 1, 1)
plt.hist(X, x)
plt.title("Data histogram")
plt.subplot(2, 1, 2)
plt.plot(xe, knn_density)
plt.title("KNN density")

# Plot KNN average relative density
plt.figure(figsize=(6, 7))
plt.subplot(2, 1, 1)
plt.hist(X, x)
plt.title("Data histogram")
plt.subplot(2, 1, 2)
plt.plot(xe, knn_avg_rel_density)
plt.title("KNN average relative density")

plt.show()

print("Ran Exercise 11.2.3")
