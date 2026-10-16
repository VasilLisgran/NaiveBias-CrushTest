import numpy as np
import matplotlib.pyplot as plt

first = [-1, -1]
second = [1, 1]

Var = 1
Cov = [[1, 0], [0, 1]]

rng = np.random.Generator(np.random.PCG64(12345))
samples_first = rng.multivariate_normal(mean=first, cov=Cov, size=1000)
sample_second = rng.multivariate_normal(mean=second, cov=Cov, size=1000)

x_first = samples_first[:, 0]
y_first = samples_first[:, 1]

x_second = sample_second[:, 0]
y_second = sample_second[:, 1]


fig, axes = plt.subplots(2)
axes[0].scatter(x_first, y_first)
axes[0].set_aspect('equal', adjustable='box')
axes[1].scatter(x_second, y_second)
axes[1].set_aspect('equal', adjustable='box')
plt.show()