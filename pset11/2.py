import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from veh_speed_sched_data import n, a, b, c, d, smin, smax, tau_min, tau_max
#fixing data
d = np.asarray(d)
smin = np.asarray(smin)
smax = np.asarray(smax)
tau_min = np.asarray(tau_min)
tau_max = np.asarray(tau_max)

d = d[:,0]
smin = smin[:,0]
smax = smax[:,0]
tau_min = tau_min[:,0]
tau_max = tau_max[:,0]

k = cp.Variable(n)
h = cp.cumsum(k)
phi = a * cp.multiply(cp.inv_pos(k), d**2) + c * k + cp.multiply(b, d)
obj = cp.Minimize(cp.sum(phi))
constraints = [
        cp.multiply(smin, k) <= d,
        cp.multiply(smax, k) >= d,
        tau_min <= h,
        tau_max >= h,
        ]
problem = cp.Problem(obj, constraints)
problem.solve()
print(f"status: {problem.status}")
if problem.status == 'optimal':
    print(f"Total fuel: {problem.value}")
    s = d / k.value
    plt.step(np.arange(n), s)
    plt.show()
