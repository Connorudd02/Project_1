# exercise 3.1.2
# (requires data structures from ex. 3.1.1)
# Imports the numpy and xlrd package, then runs the ex3_1_1 code
from ex3_1_1 import *
import matplotlib.pyplot as plt

# Data attributes to be plotted
i = 0
j = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
plt.plot(X[:, i], X[:, j], "o")

##
# Make another more fancy plot that includes legend, class labels,
# attribute names, and a title.
f = plt.figure()
plt.title("NanoNose data")

for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c 
    plt.plot(X[class_mask, i], X[class_mask, j], "o", alpha=0.3)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])

# Output result to screen
plt.show()
print("Ran Exercise 3.1.2")
