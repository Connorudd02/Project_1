# ex12_1_5
# Load data from the wine dataset
import importlib_resources
from scipy.io import loadmat

from dtuimldmtools import binarize2

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine.mat")

mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
attributeNames = [name[0][0] for name in mat_data["attributeNames"]]

# We will now transform the wine dataset into a binary format. Notice the changed attribute names:


Xbin, attributeNamesBin = binarize2(X, attributeNames)
print("X, i.e. the wine dataset, has now been transformed into:")
print(Xbin)
print(attributeNamesBin)
