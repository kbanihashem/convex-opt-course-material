import numpy as np
import cvxpy as cp

np.set_printoptions(precision=4, suppress=True)
n = 4
A_tot = 10000
alpha = np.array([1e-5, 1e-2, 1e-2, 1e-2])
M = np.array([0.1, 5, 10, 10])
A_max = np.array([40, 40, 40, 20])

b = cp.Variable(n)
r = cp.Variable(n)
s = cp.Variable(n)
beta = np.log(alpha)
m = np.log(M)
B = np.log(A_max)

constraints = [
        b <= B,
        cp.sum(b) == np.log(A_tot),
        s <= m,
        s[1:] == s[:-1] + b[1:],
        r[1:] >= b[1:] + beta[1:] + cp.logistic(2 * r[:-1] - 2 * beta[1:]),
        r[0] == beta[0] + b[0],
        ]
obj = cp.Maximize(s[-1] - r[-1])
problem = cp.Problem(obj, constraints)
problem.solve()
a = np.exp(b.value)
print(f"optimal dynamic range = {np.exp(problem.value)}")
print(f"optimal gain = {a}")
