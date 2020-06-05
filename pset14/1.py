import numpy as np
import cvxpy as cp
from scipy.linalg import solve_banded
from time import time
np.random.seed(78)
np.set_printoptions(precision=4)

delta = 1
eta = 1
n = 4000
k = 100

Y = np.random.randn(n, k)
b = np.random.randn(k)
c = Y.dot(b)
X = np.identity(n) * eta
X[np.arange(n - 1), np.arange(n - 1) + 1] -= delta
X[np.arange(n - 1) + 1, np.arange(n - 1)] -= delta
X[np.arange(n - 1), np.arange(n - 1)] += delta
X[np.arange(n - 1) + 1, np.arange(n - 1) + 1] += delta

def naive():
    A = X + Y.dot(Y.T)
    x = np.linalg.solve(A, c)
    return x

def smart():
    X = np.zeros((3, n))
    X[0][1:] -= delta
    X[2][:-1] -= delta
    X[1][:-1] += delta
    X[1][1:] += delta
    X[1] += eta
    q = np.identity(k) + np.dot(Y.T, solve_banded((1, 1), X, Y))
    y = np.linalg.solve(q, np.dot(Y.T, solve_banded((1, 1), X, c)))
    x = solve_banded((1, 1), X, c - np.dot(Y, y))
    return x

def main():
    t0 = time()
    x_naive = naive()
    t1 = time()
    t_naive = t1 - t0

    t0 = time()
    x_smart = smart()
    t1 = time()
    t_smart = t1 - t0

    l2 = np.linalg.norm
    print(f"l2(diff) / l2(x_naive) = {l2(x_smart - x_naive) / l2(x_naive)}")
    print(f"Time for naive: {t_naive}")
    print(f"Time for smart: {t_smart}")
    print(f"Time ratio: {t_naive / t_smart:.2f}")

main()
