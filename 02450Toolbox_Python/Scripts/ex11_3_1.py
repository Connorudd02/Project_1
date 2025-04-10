# exercise 11.3.1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

# Draw samples from mixture of gaussians (as in exercise 11.1.1), add outlier
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
X[-1, 0] = -10  # added outlier


# Compute kernel density estimate
kde = gaussian_kde(X.ravel())

scores = kde.evaluate(X.ravel())
idx = scores.argsort()
scores.sort()

print("The index of the lowest density object: {0}".format(idx[0]))

# Plot kernel density estimate
plt.figure()
plt.bar(range(20), scores[:20])
plt.title("Outlier score")
plt.show()

print("Ran Exercise 11.3.1")
