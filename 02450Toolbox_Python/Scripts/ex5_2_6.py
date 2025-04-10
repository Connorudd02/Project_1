# exercise 5.2.6
import sklearn.linear_model as lm

# requires data from exercise 5.1.4
from ex5_1_5 import *
import matplotlib.pyplot as plt

# Fit logistic regression model

model = lm.LogisticRegression()
model = model.fit(X, y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0]

# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([6.9, 1.09, 0.06, 2.1, 0.0061, 12, 31, 0.99, 3.5, 0.44, 12]).reshape(1, -1)
# Evaluate the probability of x being a white wine (class=0)
x_class = model.predict_proba(x)[0, 0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print("\nProbability of given sample being a white wine: {0:.4f}".format(x_class))
print("\nOverall misclassification rate: {0:.3f}".format(misclass_rate))

f = plt.figure()
class0_ids = np.nonzero(y == 0)[0].tolist()
plt.plot(class0_ids, y_est_white_prob[class0_ids], ".y")
class1_ids = np.nonzero(y == 1)[0].tolist()
plt.plot(class1_ids, y_est_white_prob[class1_ids], ".r")
plt.xlabel("Data object (wine sample)")
plt.ylabel("Predicted prob. of class White")
plt.legend(["White", "Red"])
plt.ylim(-0.01, 1.5)

plt.show()

print("Ran Exercise 5.2.6")
