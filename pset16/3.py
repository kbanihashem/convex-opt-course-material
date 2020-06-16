import numpy as np
import cvxpy as cp
from data.blend_design_data import n, k, W, W_min, W_max, P, D, A, P_spec, D_spec, A_spec
np.set_printoptions(precision=6, suppress=True)

#fixing the data
W = np.asarray(W)
A = np.asarray(A)
P = np.asarray(P)
D = np.asarray(D)

lb = cp.Variable(k) #lb is short for lambda
data = [
        (A, A_spec),
        (P, P_spec),
        (D, D_spec),
        ]
constraints = [np.log(F) @ lb <= np.log(F_spec) for (F, F_spec) in data]
constraints += [
        lb >= 0,
        cp.sum(lb) == 1,
        ]
obj = cp.Minimize(0)
problem = cp.Problem(obj, constraints)
problem.solve()
print(f'Problem status: {problem.status}')
print(f'optim lb: {lb.value}')

V = np.log(W)
v = V @ lb.value
w = np.exp(v)
print(f'feasible w: {w}')
