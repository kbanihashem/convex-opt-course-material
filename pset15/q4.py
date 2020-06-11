import numpy as np
import cvxpy as cp

np.set_printoptions(precision=8, suppress=True)

pi = np.array([1/3, 1/6, 1/3, 1/6])
R = np.array([
    [2, 2, 0.5, 0.5],
    [1.3, 0.5, 1.3, 0.5],
    [1, 1, 1, 1]
    ])
n = 3
x = cp.Variable(n)
constraints = [cp.sum(x) == 1]
obj = cp.Maximize(pi @ cp.log(R.T @ x))
problem = cp.Problem(obj, constraints)
problem.solve()
print(f"optimal x: {x.value}")
print(f"optimal value: {problem.value:.8f}")
other_strategies = [
        [1, 0, 0],
        [0, 1, 0],
        [1/2, 1/2, 0],
        ]
for st in other_strategies:
    print(f"Strategy : {st}")
    x.value = np.array(st)
    print(f"Strategy value: {obj.value:.8f}")
