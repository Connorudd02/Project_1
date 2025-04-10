## exercise 0.5.1
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 1, 0.1)
f = np.exp(x)


plt.figure(1)
plt.plot(x, f)
plt.xlabel("x")
plt.ylabel("f(x)=exp(x)")
plt.title("The exponential function")
plt.show()

# %%
