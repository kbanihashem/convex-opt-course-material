import numpy as np
import cvxpy as cp
from data.multi_risk_portfolio_data import (n, M, gamma, mu,
        Sigma_1,
        Sigma_2,
        Sigma_3,
        Sigma_4,
        Sigma_5,
        Sigma_6,
        )
np.set_printoptions(precision=4, suppress=True)

Sigmas = [
        Sigma_1,
        Sigma_2,
        Sigma_3,
        Sigma_4,
        Sigma_5,
        Sigma_6,
        ]

w = cp.Variable(n)
t = cp.Variable()
constraints = [t >= cp.quad_form(w, Sigmas[i]) for i in range(M)]
constraints.append(cp.sum(w) == 1)
obj = cp.Maximize(mu @ w - gamma * t)
problem = cp.Problem(obj, constraints)
problem.solve()
gammas = np.array([constraints[i].dual_value for i in range(M)])
print('status: ', problem.status)
print('gammas: ', gammas)
print('w: ', w.value)
print('problem value', problem.value)
