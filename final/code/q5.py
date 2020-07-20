import numpy as np
import cvxpy as cp

s = cp.Variable()
l = cp.Variable()
w = cp.quad_over_lin(s, l)
constraints = [
        s <= l,
        l <= s * np.sqrt(2),
        20 <= l,
        l <= 30,
        w <= 20,
        s >= np.sqrt(300),
        ]
obj = cp.Minimize(2 * s**2 + 2 * l + np.pi * w)
problem = cp.Problem(obj, constraints)
problem.solve(solver=cp.ECOS)
print('status: ', problem.status)
print('total cost: ', problem.value)
print('l: ', l.value)
print('w: ', w.value)
print('filter size: ', s.value**2)

