import numpy as np
import cvxpy as cp
from rank_one_nmf_data import m, n, p, Omega, A

xp = cp.Variable(m)
yp = cp.Variable(n)
err = 0
for i in range(m):
    for j in range(n):
        if Omega[i, j]:
            err += cp.exp(cp.abs(np.log(A[i, j]) - xp[i] - yp[j]))
obj = cp.Minimize(err)
constraints = []
problem = cp.Problem(obj, constraints)
problem.solve()

x = np.exp(xp.value)
y = np.exp(yp.value)
full_A = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        if Omega[i, j]:
            full_A[i, j] = A[i, j]
        else:
            full_A[i, j] = x[i] * y[j]

y *= x.sum()
x /= x.sum()
