import numpy as np
import cvxpy as cp

from data.currency_exchange_data import n, F, tickers, data, c_req, c_init
np.set_printoptions(precision=2, suppress=True)

value = np.zeros(n)
for i in range(n):
    value[i] = np.sqrt(F[i, 0] / F[0, i])
X = cp.Variable((n, n))
after_we_pay = c_init - cp.sum(X, axis=0)
c = after_we_pay + cp.sum(cp.multiply(X, 1/F), axis=1)
cost = value @ (c_init - c)
constraints = [
        c >= c_req,
        after_we_pay >= 0,
        X >= 0,
        ]
obj = cp.Minimize(cost)
problem = cp.Problem(obj, constraints)
problem.solve()
print(f"value: {value}")
print(f"problem status: {problem.status}")
print(f"optimal cost: {problem.value}")
print(f"optimal X:\n {X.value}")
print(f"Initial holding (c_init): {c_init}")
print(f"Required holding (c_req): {c_req}")
print(f"Final holding: {c.value}")
