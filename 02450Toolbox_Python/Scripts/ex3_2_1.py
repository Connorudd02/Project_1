## exercise 3.2.1
import importlib_resources
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat

filename = importlib_resources.files("dtuimldmtools").joinpath("data/zipdata.mat")
# Index of the digit to display
i = 0

# Load Matlab data file to python dict structure
mat_data = loadmat(filename)

# Extract variables of interest
testdata = mat_data["testdata"]
traindata = mat_data["traindata"]
X = traindata[:, 1:]
y = traindata[:, 0]


# Visualize the i'th digit as a vector
f = plt.figure()
plt.subplot(4, 1, 4)
plt.imshow(np.expand_dims(X[i, :], axis=0), extent=(0, 256, 0, 10), cmap=plt.cm.gray_r)
plt.xlabel("Pixel number")
plt.title("Digit in vector format")
plt.yticks([])

# Visualize the i'th digit as an image
plt.subplot(2, 1, 1)
I = np.reshape(X[i, :], (16, 16))
plt.imshow(I, extent=(0, 16, 0, 16), cmap=plt.cm.gray_r)
plt.title("Digit as an image")

plt.show()


