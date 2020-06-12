import numpy as np
import cvxpy as cp
from data.opt_funding_data import n, T, rp, rn, E, C, P, M, A
np.set_printoptions(precision=6, suppress=True)

x = cp.Variable(n)
E = [0] + list(E)
I = [0] + list(A @ x)
B = [None] * (T + 1)
B[0] = cp.Variable()
for t in range(T):
    before = B[t] - E[t] + I[t]
    B[t + 1] = cp.minimum((1 + rp) * before, (1 + rn) * before)
#B = cp.hstack(B)
constraints = [
        B[-1] + I[-1] - E[-1] >= 0,
        B[0] >= 0,
        x >= 0,
        ]

obj = cp.Minimize(x @ P + B[0])
problem = cp.Problem(obj, constraints)
"""
problem.solve()
"""
