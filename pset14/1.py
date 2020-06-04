import numpy as np
import cvxpy as cp
from scipy.linalg import solve_banded
from time import time

delta = 1
eta = 1
n = 4000
k = 100
np.random.seed(78)
Y = np.random.randn(n, k)
X = np.identity(n) * eta
X[np.arange(n - 1), np.arange(n - 1) + 1] -= delta
X[np.arange(n - 1) + 1, np.arange(n - 1)] -= delta
X[np.arange(n - 1), np.arange(n - 1)] += delta
X[np.arange(n - 1) + 1, np.arange(n - 1) + 1] += delta
print(X)

def naive():
    A = X + Y.dot(Y.T)
    x = np.linalg.solve(A, b)
    return x

def smart():
    pass

