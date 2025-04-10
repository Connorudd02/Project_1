# exercise 4.1.2

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


# Compute empirical mean and standard deviation
mu_ = X.mean()
s_ = X.std(ddof=1)

print("Theoretical mean: ", mu)
print("Theoretical std.dev.: ", s)
print("Empirical mean: ", mu_)
print("Empirical std.dev.: ", s_)

# Plot the samples and histogram
plt.figure()
plt.title("Normal distribution")
plt.subplot(1, 2, 1)
plt.plot(X, "x")
plt.subplot(1, 2, 2)
plt.hist(X, bins=nbins)
plt.show()

