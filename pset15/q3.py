import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data.min_time_speed_data import N, m, d, h, eta, C_D, P, F, g
np.set_printoptions(precision=4, suppress=True)

experiment_names = ['optimal', 'uniform']
for name in experiment_names:
    print(f"starting experiment {name}")
    l = cp.Variable(N + 1)
    f = cp.Variable(N + 1)
    T = cp.sum(d * cp.inv_pos(cp.sqrt(l))[:-1])
    constraints = [
            m * l[1:] / 2 + m * g * h[1:] == m * l[:-1]/2 + m * g * h[:-1] + eta * f[1:] - d * C_D * l[:-1],
            eta * f[0] == m * l[0] / 2,
            T * P / eta + cp.sum(f) <= F,
            l[-1] >= 0,
            ]
    if name == 'uniform':
        constraints.append(f[1:] == f[:-1])

    obj = cp.Minimize(T)
    problem = cp.Problem(obj, constraints)
    problem.solve()
    print(f"problem status: {problem.status}")
    print(f"problem.value: {problem.value}")
    print(f"optimal f: {f.value}")
    s = np.sqrt(l.value)
    print(f"optimal speeds: {s}")
    for i in range(N):
        distance = [i * d, (i + 1) * d]
        speed = [s[i]] * 2 
        plt.plot(distance, speed, color='red')
    plt.show()
