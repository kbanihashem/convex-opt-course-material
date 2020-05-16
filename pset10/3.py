import numpy as np
import cvxpy as cp

from satisfy_some_constraints_data import m, n, k, A, b, c
epsilon = 1e-5

u = cp.Variable()
v = cp.Variable()
x = cp.Variable((A.shape[1],))
f_x = A @ x - b
constraints = [
        cp.sum(cp.pos(f_x + u)) <= (m - k) * u,
        cp.inv_pos(v) <= u,
        ]
obj = cp.Minimize(c @ x)
p1 = cp.Problem(obj, constraints)
p1.solve()
if p1.status == 'optimal':
    lambda_value = 1 / u.value
    print(f'lambda: {lambda_value:.4f}')
    print(f'objective value: {p1.value:.4f}')
    f_values = np.matmul(A, x.value) - b
    satisfied = f_values <= epsilon
    satisfied_count = satisfied.sum()
    print(f"satisfied count: {satisfied_count}")
    #part b
    smallest_indexes = np.argsort(f_values)[:k]
    constraints = [A[smallest_indexes,:] @ x <= b[smallest_indexes]]
    p2 = cp.Problem(obj, constraints) 
    p2.solve()
    print(f'Improved objective value: {p2.value:.4f}')
