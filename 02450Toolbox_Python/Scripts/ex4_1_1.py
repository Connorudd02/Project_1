# exercise 4.1.1
import numpy as np
import matplotlib.pyplot as plt 

# Number of samples
N = 200

# Mean
mu = 17

# Standard deviation
s = 2

# Number of bins in histogram
nbins = 20

# Generate samples from the Normal distribution
X = np.random.normal(mu, s, N).T
# or equally:
X = np.random.randn(N).T * s + mu

# Plot the samples and histogram
plt.figure(figsize=(12, 4))
plt.title("Normal distribution")
plt.subplot(1, 2, 1)
plt.plot(X, ".")
plt.subplot(1, 3, 3)
plt.hist(X, bins=nbins)
plt.show()
