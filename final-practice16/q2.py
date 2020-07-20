import numpy as np
import cvxpy as cp
from data.correlation_bounds_data import m, n, A, sigma
np.set_printoptions(precision=4, suppress=True)

Sigma = cp.Variable((n, n), PSD=True)
constraints = []
for i in range(m):
    a = A[:,i]
    constraints.append(cp.quad_form(a, Sigma) == sigma[i] ** 2)

rhos = []
for i in range(n):
    for j in range(i):
        denom = cp.geo_mean(cp.hstack([Sigma[i, i], Sigma[j, j]]))
        rho_ij = cp.quad_over_lin(Sigma[i, j], denom)
        rhos.append(rho_ij)

rho_max = cp.max(cp.hstack(rhos))
obj = cp.Minimize(rho_max)
problem = cp.Problem(obj, constraints)
problem.solve()
print(problem.status)
print(Sigma.value)
print(rho_max.value)
