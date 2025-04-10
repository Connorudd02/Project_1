# exercise 2.4.1
"""
Note: This is a long script. You may want to use breakpoints 
"""
import importlib_resources
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import zscore

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
C = mat_data["C"][0, 0]
M = mat_data["M"][0, 0]
N = mat_data["N"][0, 0]
attributeNames = [name[0][0] for name in mat_data["attributeNames"]]
classNames = [cls[0][0] for cls in mat_data["classNames"]]

print("Data loaded")

# We start with a box plot of each attribute
plt.figure()
plt.title("Wine: Boxplot")
plt.boxplot(X)
plt.xticks(range(1, M + 1), attributeNames, rotation=45)

# From this it is clear that there are some outliers in the Alcohol
# attribute (10x10^14 is clearly not a proper value for alcohol content)
# However, it is impossible to see the distribution of the data, because
# the axis is dominated by these extreme outliers. To avoid this, we plot a
# box plot of standardized data (using the zscore function).
plt.figure(figsize=(12, 6))
plt.title("Wine: Boxplot (standarized)")
plt.boxplot(zscore(X, ddof=1), attributeNames)
plt.xticks(range(1, M + 1), attributeNames, rotation=45)

# This plot reveals that there are clearly some outliers in the Volatile
# acidity, Density, and Alcohol attributes, i.e. attribute number 2, 8,
# and 11.
plt.show()

# Next, we plot histograms of all attributes.
plt.figure(figsize=(14, 9))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N)  # Make the y-axes equal for improved readability
    if i % v != 0:
        plt.yticks([])
    if i == 0:
        plt.title("Wine: Histogram")

plt.show()

# This confirms our belief about outliers in attributes 2, 8, and 11.
# To take a closer look at this, we next plot histograms of the
# attributes we suspect contains outliers
plt.figure(figsize=(14, 9))
m = [1, 7, 10]
for i in range(len(m)):
    plt.subplot(1, len(m), i + 1)
    plt.hist(X[:, m[i]], 50)
    plt.xlabel(attributeNames[m[i]])
    plt.ylim(0, N)  # Make the y-axes equal for improved readability
    if i > 0:
        plt.yticks([])
    if i == 0:
        plt.title("Wine: Histogram (selected attributes)")

plt.show()

# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.
outlier_mask = (X[:, 1] > 20) | (X[:, 7] > 10) | (X[:, 10] > 200)
valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
X = X[valid_mask, :]
y = y[valid_mask]
N = len(y)

# Now, we can repeat the process to see if there are any more outliers
# present in the data. We take a look at a histogram of all attributes:
plt.figure(figsize=(14, 9))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N)  # Make the y-axes equal for improved readability
    if i % v != 0:
        plt.yticks([])
    if i == 0:
        plt.title("Wine: Histogram (after outlier detection)")

# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

plt.show()

print("Ran Exercise 2.4.1")
