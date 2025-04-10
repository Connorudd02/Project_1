# exercise 6.3.1
import importlib_resources
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth3.mat")  # <-- change the number to change dataset

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
X_train = mat_data["X_train"]
X_test = mat_data["X_test"]
y = mat_data["y"].squeeze()
y_train = mat_data["y_train"].squeeze()
y_test = mat_data["y_test"].squeeze()
attributeNames = [name[0] for name in mat_data["attributeNames"].squeeze()]
classNames = [name[0][0] for name in mat_data["classNames"]]
N, M = X.shape
C = len(classNames)


# Plot the training data points (color-coded) and test data points.
plt.figure(1)
styles = [".b", ".r", ".g", ".y"]
for c in range(C):
    class_mask = y_train == c
    plt.plot(X_train[class_mask, 0], X_train[class_mask, 1], styles[c])


# K-nearest neighbors
K = 5

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = 2
metric = "minkowski"
metric_params = {}  # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
# metric = 'cosine'
# metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
# metric='mahalanobis'
# metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(
    n_neighbors=K, p=dist, metric=metric, metric_params=metric_params
)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ["ob", "or", "og", "oy"]
for c in range(C):
    class_mask = y_est == c
    plt.plot(X_test[class_mask, 0], X_test[class_mask, 1], styles[c], markersize=10)
    plt.plot(X_test[class_mask, 0], X_test[class_mask, 1], "kx", markersize=8)
plt.title("Synthetic data classification - KNN")

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est)
accuracy = 100 * cm.diagonal().sum() / cm.sum()
error_rate = 100 - accuracy
plt.figure(2)
plt.imshow(cm, cmap="binary", interpolation="None")
plt.colorbar()
plt.xticks(range(C))
plt.yticks(range(C))
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title(
    "Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)".format(accuracy, error_rate)
)

plt.show()

print("Ran Exercise 6.3.1")
