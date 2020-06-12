import numpy as np
import cvxpy as cp
from data.opt_funding_data import n, T, rp, rn, E, C, P, M, A
np.set_printoptions(precision=6, suppress=True)

B = cp.Variable(T + 1)
I = cp.Variable(T + 1)
x = cp.Variable(n)
E = np.hstack([0, E])

constraints = [
        B[-1] + I[-1] - E[-1] == 0,
#        B[1:] <= (1 + rp) * (B[:-1] - E[:-1] + I[:-1]) - (rn - rp) * cp.neg(B[:-1] - E[:-1] + I[:-1]),
        B[:-1] >= B[1:] * (1/(1 + rn)) + cp.pos(B[1:]) * (1/(1 + rp) - 1/(1 + rn)) - I[:-1] + E[:-1],
        B[0] >= 0,
        I[0] == 0,
        I[1:] == A @ x,
        x >= 0,
        ]

obj = cp.Minimize(x @ P + B[0])
problem = cp.Problem(obj, constraints)
problem.solve()
