import numpy as np
import cvxpy as cp
lp = cp.Variable()
wp = cp.Variable()
constraints = [
        0 <= lp - wp,
        lp - wp <= np.log(2),
        np.log(20) <= lp,
        lp <= np.log(30),
        np.log(10) <= wp,
        wp <= np.log(20),
        lp + wp >= np.log(300),
        ]
cost = 2 * cp.exp(lp + wp) + 2 * cp.exp(lp) + np.pi * cp.exp(wp)
obj = cp.Minimize(cost)
problem = cp.Problem(obj, constraints)
problem.solve(solver=cp.SCS)
print('status: ', problem.status)
print('total cost: ', problem.value)
l = np.exp(lp.value)
w = np.exp(wp.value)
print('l: ', l)
print('w: ', w)
s = l * w
print('filter size: ', s)
