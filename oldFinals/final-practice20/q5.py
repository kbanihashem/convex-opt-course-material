import numpy as np
import cvxpy as cp
from data.ranked_lists_inconsistent_data import n, m, k, Sigma
#from data.ranked_lists_data import n, m, k, Sigma

def k_eval(sigma, s):
    total = 0
    for row in sigma:
        fine = True
        for i in range(len(row) - 1):
            if s[row[i]] <= s[row[i + 1]]:
                fine = False
        total += fine
    return total

sigma = Sigma.T.copy()
sigma -= 1
s = cp.Variable(n)
t = cp.Variable(m)
#t = np.ones(m)
constraints = [
        t <= 1,
        ]
for j, row in enumerate(sigma):
    for i in range(len(row) - 1):
        constraints.append(
                s[row[i]] >= s[row[i + 1]] + t[j]
                )
obj = cp.sum(t)
problem = cp.Problem(cp.Maximize(obj), constraints)
problem.solve()
print(problem.status)
ordering = np.argsort(-s.value)
ans = k_eval(sigma, s.value)
print(ans)
