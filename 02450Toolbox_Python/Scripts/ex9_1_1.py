# exercise 9.1.1
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression

from dtuimldmtools import BinClassifierEnsemble, bootstrap, dbplot, dbprobplot

filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth5.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# Fit model using bootstrap aggregation (bagging):

# Number of rounds of bagging
L = 100

# Weights for selecting samples in each bootstrap
weights = np.ones((N,1),dtype=float)/N

# Storage of trained log.reg. classifiers fitted in each bootstrap
logits = [0]*L
votes = np.zeros((N,))

# For each round of bagging
for l in range(L):

    # Extract training set by random sampling with replacement from X and y
    X_train, y_train = bootstrap(X, y, N, weights)
    
    # Fit logistic regression model to training data and save result
    logit_classifier = LogisticRegression()
    logit_classifier.fit(X_train, y_train)
    logits[l] = logit_classifier
    y_est = logit_classifier.predict(X).T
    votes = votes + y_est

    ErrorRate = (y!=y_est).sum(dtype=float)/N
    print('Error rate: {:2.2f}%'.format(ErrorRate*100))    
    
# Estimated value of class labels (using 0.5 as threshold) by majority voting
y_est_ensemble = votes>(L/2)

# Compute error rate
ErrorRate = (y!=y_est_ensemble).sum(dtype=float)/N
print('Error rate: {:3.2f}%'.format(ErrorRate*100))

ce = BinClassifierEnsemble(logits)
plt.figure(1); dbprobplot(ce, X, y, 'auto', resolution=200)
plt.figure(2); dbplot(ce, X, y, 'auto', resolution=200)

plt.show()

print('Ran Exercise 9.1.1')