import numpy as np
import cvxpy as cp

from data.team_data import n, m, m_test, sigma, train, test
np.set_printoptions(precision=4)
#b
print(n, m)
a = cp.Variable(n)
g = [None] * m
for i, (j, k, y) in enumerate(train):
    if y == -1:
        j, k = k, j
    j -= 1
    k -= 1
    Aa = a[j] - a[k]
    g[i] = cp.log_sum_exp(cp.vstack([0, -2 * Aa / sigma]))
g = cp.vstack(g)

obj = cp.Minimize(cp.sum(g))
constraints = [
        a >= 0,
        a <= 1,
        ]
problem = cp.Problem(obj, constraints) 
problem.solve()
print(f"problem status: {problem.status}")
print(f"a: {a.value}")
print(f"exp(mean(logprobs)): {np.exp(-problem.value / m)}")
#c
log_mine = []
log_dumb_algorithm = []
for i, (j, k, y) in enumerate(test):
    my_pred = 1 if (a.value[j - 1] - a.value[k - 1]) > 0 else -1
    #just making sure that data is the same
    assert tuple(train[i][:-1]) == (j, k)
    dumb_pred = train[i][-1]

    log_mine.append(my_pred == y)
    log_dumb_algorithm.append(dumb_pred == y)

print(f"Our algorithm success rate: {np.mean(log_mine)}")
print(f"Dumb algorithm success rate: {np.mean(log_dumb_algorithm)}")
