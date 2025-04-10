# Exercise 2.3.5
# (requires data from exercise 2.3.1)
from ex2_3_1 import *
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = y == c
            plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".")
            if m1 == M - 1:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])
                            
plt.legend(classNames)

plt.show()

print("Ran Exercise 2.3.5")
