import numpy as np
import cvxpy as cp

n = 3
u = 2
gamma = 2
G = (np.ones((n, n)) - np.identity(n)) / 2
d = cp.Variable(n, nonneg=True)
k = cp.Variable(n, nonneg=True)

