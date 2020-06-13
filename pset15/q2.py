import numpy as np
import cvxpy as cp
np.set_printoptions(precision=4, suppress=True)

n = 3
G = np.array([
    [0.3, -0.1, -0.9],
    [-0.6, 0.3, -0.3],
    [-0.3, 0.6, 0.2],
    ])

K = cp.Variable((n, n))
Z = cp.Variable((n, n))
constraints = [
        np.identity(n) + G @ Z == K,
        cp.norm(Z, axis=1) <= 1,
        ]
obj = cp.Minimize(cp.max(cp.norm(K, axis=1)))
problem = cp.Problem(obj, constraints)
problem.solve()
print("Optimal value: ", problem.value)
F = Z.value.dot(np.linalg.inv(K.value))
print(f"F: {F}")
