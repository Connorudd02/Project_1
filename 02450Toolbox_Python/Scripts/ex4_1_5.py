# exercise 4.1.5

import numpy as np
import matplotlib.pyplot as plt 

# Number of samples
N = 1000

# Standard deviation of x1
s1 = 2

# Standard deviation of x2
s2 = 3

# Correlation between x1 and x2
corr = 0.5

# Covariance matrix
S = np.matrix([[s1 * s1, corr * s1 * s2], [corr * s1 * s2, s2 * s2]])

# Mean
mu = np.array([13, 17])

# Number of bins in histogram
nbins = 20

# Generate samples from multivariate normal distribution
X = np.random.multivariate_normal(mu, S, N)


# Plot scatter plot of data
plt.figure(figsize=(12, 8))
plt.suptitle("2-D Normal distribution")

plt.subplot(1, 2, 1)
plt.plot(X[:, 0], X[:, 1], "x")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Scatter plot of data")

plt.subplot(1, 2, 2)
x = np.histogram2d(X[:, 0], X[:, 1], nbins)
plt.imshow(x[0], cmap=plt.cm.gray_r, interpolation="None", origin="lower")
plt.colorbar()
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([])
plt.yticks([])
plt.title("2D histogram")

plt.show()


