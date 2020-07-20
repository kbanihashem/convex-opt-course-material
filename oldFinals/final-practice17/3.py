import numpy as np
import cvxpy as cp
from path_plan_contingencies_data import N, O, S, h, p1, p2, pN_1, pN
import matplotlib.pyplot as plt
p = cp.Variable((3, N))
P = p.T
constraints = [
        p[0,:S] == p[1,:S],
        p[0,:S] == p[2,:S],
        p >= -1,
        p <= 1,
        p[:,0] == p1,
        p[:,1] == p2,
        p[:,-2] == pN_1,
        p[:,-1] == pN,
        p[0, O] >= 0.5,
        p[1, O] <= -0.5,
        ]

penalty = cp.sum((p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:])**2)
obj = cp.Minimize(penalty)
problem = cp.Problem(obj, constraints)
problem.solve()
print(problem.value)

def plot():
    ### FEEL FREE TO COPY THE FOLLOWING PLOTTING CODE ###
    fig, ax = plt.subplots()
    x = np.arange(1, N + 1, 1) * h
    plt.plot((S * h, S * h), (-1, 1), 'k--', label='obstacle revealed')
    ax.set_xlim((x[0], x[N-1])); ax.set_ylim((-1.2, 1.2))
    ax.plot(x, (P.value)[:, 0], color='red', label='obstacle left')
    ax.plot(x, (P.value)[:, 1], color='blue', label='obstacle right')
    ax.plot(x, (P.value)[:, 2], color='green', label='obstacle clear')
    ax.plot(x, [1]*N, color='black')
    ax.plot(x, [-1]*N, color='black')
    plt.plot((O * h, O * h), (0.5, 1), 'r--', label='left safe region')
    plt.plot((O * h, O * h), (-1, -0.5), 'b--', label='right safe region')
    plt.legend(loc = 4)
    plt.show(); fig.savefig('path_plan_contingencies.eps')

plot()
