import numpy as np
import cvxpy as cp
from data.clearing_data import n, L1, c1
def check_feas(T):
    P = [cp.Variable((n, n), pos=True) for _ in range(T - 1)]
    L = [cp.Variable((n, n), pos=True) for _ in range(T)]
    c = [cp.Variable(n) for _ in range(T)]
    constraints = [
            L[-1] == 0,
            c[0] == c1,
            L[0] == L1,
            ]
    for t in range(T - 1):
        constraints += [
                L[t][np.diag_indices(n)] == 0,
                L[t + 1] == L[t] - P[t],
                c[t + 1] == c[t] - cp.sum(P[t], axis=1) + cp.sum(P[t], axis=0),
                cp.sum(P[t], axis=1) <= c[t],
                ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve()
    print(list(map(lambda x: x.value, c)))
    return problem.status

#N = 20
#for i in range(1, N):
#    res = check_feas(i)
#    print(i, res)
#    if res == 'optimal':
#        break
res = check_feas(5)
print(res)
