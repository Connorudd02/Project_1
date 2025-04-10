# exercise 3.2.2
import importlib_resources
import numpy as np
from scipy.linalg import svd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat

filename = importlib_resources.files("dtuimldmtools").joinpath("data/zipdata.mat")

# Digits to include in analysis (to include all, n = range(10) )
n = [0, 1]
# Number of principal components for reconstruction
K = 16
# Digits to visualize
nD = range(6)

# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat(filename)["traindata"]
X = traindata[:, 1:]
y = traindata[:, 0]

N, M = X.shape
C = len(n)

classValues = n
classNames = [str(num) for num in n]
classDict = dict(zip(classNames, classValues))


# Select subset of digits classes to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = y == v
    # Use the logical OR operator ("|") to select rows
    # already in class_mask OR cmsk
    class_mask = class_mask | cmsk
X = X[class_mask, :]
y = y[class_mask]
N = X.shape[0]

# Center the data (subtract mean column values)
Y = X - np.ones((N, 1)) * X.mean(0) 

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False) #NOTE: Change to Vh
# U = mat(U)
V = Vh.T

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

# Project data onto principal component space
Z = Y @ V

# Plot variance explained
plt.figure()
plt.plot(rho, "o-")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained value")


# Plot PCA of the data
f = plt.figure()
plt.title("pixel vectors of handwr. digits projected on PCs")
for c in n:
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, 0], Z[class_mask, 1], "o")
plt.legend(classNames)
plt.xlabel("PC1")
plt.ylabel("PC2")


# Visualize the reconstructed data from the first K principal components
# Select randomly D digits.
plt.figure(figsize=(10, 3))
W = Z[:, range(K)] @ V[:, range(K)].T
D = len(nD)
for d in range(D):
    # Select random digit index
    digit_ix = np.random.randint(0, N)
    plt.subplot(2, D, int(d + 1))
    # Reshape the digit from vector to 16x16 array
    I = np.reshape(X[digit_ix, :], (16, 16))
    plt.imshow(I, cmap=plt.cm.gray_r)
    plt.title("Original")
    plt.subplot(2, D, D + d + 1)
    # Reshape the digit from vector to 16x16 array
    I = np.reshape(W[digit_ix, :] + X.mean(0), (16, 16))
    plt.imshow(I, cmap=plt.cm.gray_r)
    plt.title("Reconstr.")


# Visualize the pricipal components
plt.figure(figsize=(8, 6))
for k in range(K):
    N1 = int(np.ceil(np.sqrt(K)))
    N2 = int(np.ceil(K / N1))
    plt.subplot(N2, N1, int(k + 1))
    I = np.reshape(V[:, k], (16, 16))
    plt.imshow(I, cmap=plt.cm.hot)
    plt.title("PC{0}".format(k + 1))

# output to screen
plt.show()
