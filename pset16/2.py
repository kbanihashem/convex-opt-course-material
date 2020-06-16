import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from data.ml_estim_incr_signal_data import N, k, h, y
y = np.asarray(y)

expermient_names = ['positive', 'relaxed']
x_values = dict()
for name in expermient_names:
    x = cp.Variable(N)
    yhat = cp.conv(h, x)[:N]
    error = (yhat - y)**2
    obj = cp.Minimize(cp.sum(error))
    if name == 'positive':
        constraints = [
                x[0] >= 0,
                x[1:] >= x[:-1],
                ]
    else:
        constraints = []

    problem = cp.Problem(obj, constraints)
    problem.solve()
    print(f'Experiment {name}. status: {problem.status}')
    print(f'Optimal objective: {problem.value:.6f}')
    x_values[name] = x.value

from data.ml_estim_incr_signal_data import xtrue
xtrue = np.asarray(xtrue)
x_values['true'] = xtrue
fig, ax = plt.subplots()
for name, val in x_values.items():
    ax.plot(np.arange(N), val, label=name)
ax.legend()
plt.show()
