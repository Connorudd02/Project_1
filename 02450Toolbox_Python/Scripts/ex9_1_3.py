# exercise 9.1.3
import importlib_resources
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
from dtuimldmtools import dbplot, dbprobplot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth7.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)


# Number of rounds of bagging
L = 10

# Fit model using random tree classifier:
rf_classifier = RandomForestClassifier(L)
rf_classifier.fit(X, y)
y_est = rf_classifier.predict(X).T
y_est_prob = rf_classifier.predict_proba(X).T

# Compute classification error
ErrorRate = (y!=y_est).sum(dtype=float)/N
print('Error rate: {:.2f}%'.format(ErrorRate*100))    

# Plot decision boundaries    
plt.figure(1); dbprobplot(rf_classifier, X, y, 'auto', resolution=400)
plt.figure(2); dbplot(rf_classifier, X, y, 'auto', resolution=400)

plt.show()

print('Ran Exercise 9.1.3')