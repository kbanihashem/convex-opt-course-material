import numpy as np
import cvxpy as cp

n = 3
u = 2
gamma = 2
G = (np.ones((n, n)) - np.identity(n)) / 2
d = cp.Variable(n)
k = cp.Variable(n)
A = cp.Variable((n, n))
constraints = [
        d >= 0,
        k >= 0,
        A == cp.diag(d) + cp.diag(k) * G,
        gamma * cp.sum(cp.inv_pos(d)) + cp.sum(cp.inv_pos(k)) <= u,
        ]
obj = cp.Minimize(cp.lambda_max(A))
problem = cp.Problem(obj, constraints)
problem.solve()
print(problem.status)
print(problem.value)
