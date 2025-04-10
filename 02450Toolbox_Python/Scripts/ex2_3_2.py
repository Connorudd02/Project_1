# Exercise 2.3.2
import numpy as np
# (requires data from exercise 2.3.1 so will run that script first)
from ex2_3_1 import *
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 7))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i], color=(0.2, 0.8 - i * 0.2, 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)

plt.show()

print("Ran Exercise 2.3.2")
