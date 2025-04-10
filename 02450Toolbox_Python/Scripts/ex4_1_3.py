# exercise 4.1.3
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

# Number of samples
N = 500

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

# Plot the histogram
f = plt.figure()
plt.title("Normal distribution")
plt.hist(X, bins=nbins, density=True)

# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X.min(), X.max(), 1000)
pdf = stats.norm.pdf(x, loc=17, scale=2)
plt.plot(x, pdf, ".", color="red")

# Compute empirical mean and standard deviation
mu_ = X.mean()
s_ = X.std(ddof=1)

print("Theoretical mean: ", mu)
print("Theoretical std.dev.: ", s)
print("Empirical mean: ", mu_)
print("Empirical std.dev.: ", s_)

plt.show()

