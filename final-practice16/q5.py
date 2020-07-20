import numpy as np
import cvxpy as cp
from data.ideal_pref_point_data import K, n, c, d, box, plot
np.set_printoptions(precision=6, suppress=True)
c_tilde = cp.Variable(n)
constraints = []
for i, j in d:
    a = c[i] - c[j]
    b = a @ (c[i] + c[j]) / 2
    constraints.append(a @ c_tilde >= b)

for min_max in range(2):
    for i in range(n):
        func = cp.Minimize if min_max == 0 else cp.Maximize
        obj = func(c_tilde[i])
        problem = cp.Problem(obj, constraints)
        problem.solve()
        print(problem.status)
        box[i][min_max] = problem.value

box = np.array(box)
for i in range(2):
    print(box[i, 1] - box[i, 0])
