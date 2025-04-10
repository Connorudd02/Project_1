from ex7_1_1 import *

from dtuimldmtools import jeffrey_interval

# Compute the Jeffreys interval
alpha = 0.05
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:, 0], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)
