# Exercise 2.3.3
# (requires data from exercise 2.3.1)
from ex2_3_1 import *
import matplotlib.pyplot as plt

plt.figure()
plt.boxplot(X)
plt.xticks(range(1, 5), attributeNames)
plt.ylabel("cm")
plt.title("Fisher's Iris data set - boxplot")
plt.show()

print("Ran Exercise 2.3.3")
