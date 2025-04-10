# Exercise 2.3.4
# requires data from exercise 4.1.1
from ex2_3_1 import *
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
for c in range(C):
    plt.subplot(1, C, c + 1)
    class_mask = y == c  # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c

    plt.boxplot(X[class_mask, :])
    # title('Class: {0}'.format(classNames[c]))
    plt.title("Class: " + classNames[c])
    plt.xticks(
        range(1, len(attributeNames) + 1), [a[:7] for a in attributeNames], rotation=45
    )
    y_up = X.max() + (X.max() - X.min()) * 0.1
    y_down = X.min() - (X.max() - X.min()) * 0.1
    plt.ylim(y_down, y_up)

plt.show()

print("Ran Exercise 2.3.4")
