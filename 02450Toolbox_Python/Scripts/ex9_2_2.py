# exercise 9.2.2
import importlib_resources
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from dtuimldmtools import rocplot, confmatplot

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine2.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]

attribute_included = 10   # alcohol contents
X = X[:,attribute_included].reshape(-1,1)
attributeNames = attributeNames[attribute_included]
N, M = X.shape
C = len(classNames)

# K-fold crossvalidation
K = 2
CV = StratifiedKFold(K, shuffle=True)

k=0
for train_index, test_index in CV.split(X,y):
    print(train_index)
    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    logit_classifier = LogisticRegression()
    logit_classifier.fit(X_train, y_train)

    y_test_est = logit_classifier.predict(X_test).T
    p = logit_classifier.predict_proba(X_test)[:,1].T

    plt.figure(k)
    rocplot(p,y_test)

    plt.figure(k+1)
    confmatplot(y_test,y_test_est)

    k+=2
    
plt.show()    