# exercise 10.1.3
import importlib_resources
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

from dtuimldmtools import clusterval

filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth1.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# Maximum number of clusters:
K = 10

# Allocate variables:
Rand = np.zeros((K-1,))
Jaccard = np.zeros((K-1,))
NMI = np.zeros((K-1,))

for k in range(K-1):
    # run K-means clustering:
    #cls = Pycluster.kcluster(X,k+1)[0]
    centroids, cls, inertia = k_means(X,k+2)
    # compute cluster validities:
    Rand[k], Jaccard[k], NMI[k] = clusterval(y,cls)    
        
# Plot results:

plt.figure(1)
plt.title('Cluster validity')
plt.plot(np.arange(K-1)+2, Rand)
plt.plot(np.arange(K-1)+2, Jaccard)
plt.plot(np.arange(K-1)+2, NMI)
plt.legend(['Rand', 'Jaccard', 'NMI'], loc=4)
plt.show()

print('Ran Exercise 10.1.3')
