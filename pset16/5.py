import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data.zero_crossings_data import n, f_min, B, s

C = np.cos(2 * np.pi / n * np.outer(np.arange(n) + 1, f_min + np.arange(B)))
D = np.sin(2 * np.pi / n * np.outer(np.arange(n) + 1, f_min + np.arange(B)))

a = cp.Variable(B)
b = cp.Variable(B)
y_hat = cp.Variable(n)
constraints = [
        C @ a + D @ b == y_hat,
        s @ y_hat == n,
        cp.multiply(s, y_hat) >= 0,
        ]
obj = cp.Minimize(cp.norm(y_hat))
problem = cp.Problem(obj, constraints)
problem.solve()
print(f'problem status: {problem.status}')

from data.zero_crossings_data import y
relative_recovery_error = np.linalg.norm(y_hat.value - y) / np.linalg.norm(y)
print(f'Relative recovery error = {relative_recovery_error}')
print("Short sentence: The recovery is good I guess :))")

fig, ax = plt.subplots()
ax.plot(np.arange(n), y_hat.value, label='prediction')
ax.plot(np.arange(n), y, label='truth')
ax.legend()
plt.show()
