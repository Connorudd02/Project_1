# Exercise 2.3.7
# (requires data from exercise 2.3.1)
from ex2_3_1 import *
import matplotlib.pyplot as plt
from scipy.stats import zscore

X_standarized = zscore(X, ddof=1)

plt.figure(figsize=(12, 6))
plt.imshow(X_standarized, interpolation="none", aspect=(4.0 / N), cmap=plt.cm.gray) 
plt.xticks(range(4), attributeNames)
plt.xlabel("Attributes")
plt.ylabel("Data objects")
plt.title("Fisher's Iris data matrix")
plt.colorbar()

plt.show()

print("Ran Exercise 2.3.7")