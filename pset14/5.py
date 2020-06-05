import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from data.microgrid_data import N, p_pv, p_ld, C, D, Q, R_buy, R_sell

p_batt = cp.Variable(N)
p_grid = cp.Variable(N)
q = cp.Variable(N)
p_grid = p_ld - p_batt - p_pv

constraints = [
        p_batt <= D,
        p_batt >= -C,
        q >= 0,
        q <= Q,
        q[1:] == q[:-1] - p_batt[:-1]/4,
        q[0] == q[-1] - p_batt[-1] / 4,
        p_ld == p_batt + p_pv + p_grid,
        ]
obj = cp.Minimize(R_sell @ p_grid + (R_buy - R_sell) @ cp.pos(p_grid))
problem = cp.Problem(obj, constraints)
problem.solve()
print(f"problem status: {problem.status}")
time = np.arange(N)
plt.plot(time, p_grid.value, color='blue')
plt.plot(time, p_batt.value, color='red')
plt.plot(time, q.value, color='green')
plt.plot(time, p_ld, color='yellow', linestyle='--')
plt.plot(time, p_pv, color='orange', linestyle='--')
plt.show()

