import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from data.storage_tradeoff_data import T, t, p, u
p = p.reshape(T)
u = u.reshape(T)

def solve(Q, C, D):
    q1 = cp.Variable(1)
    c = cp.Variable(T)
    q = cp.hstack([q1, cp.cumsum(c)[:-1]])

    constraints = [
            q <= Q,
            q >= 0,
            c <= C,
            c >= -D,
            q[-1] + c[-1] == q[0],
            u + c >= 0,
            ]

    obj = cp.Minimize(p @ (u + c))
    problem = cp.Problem(obj, constraints)
    problem.solve()
    return q.value, c.value, problem.value

def kplot(t, p, u, q, c):
    plt.plot(t, p, 'g')
    plt.plot(t, u, 'r')
    plt.plot(t, q, 'b')
    plt.plot(t, c, 'y')
    plt.show()

def a():
    Q = 35
    C = 3
    D = 3
    q, c, _ = solve(35, 3, 3)
    kplot(t, p, u, q, c)

def b():
    n = 150
#    Qs = np.linspace(0, 150, n)
    Qs = np.arange(n)
    for i, cd_limit in enumerate([1, 3]):
        C = cd_limit
        D = cd_limit
        cost = np.vectorize(lambda Q: solve(Q, C, D)[-1])(Qs)
        print(cost[0:5])
        color = 'r' if i == 0 else 'g'
        plt.plot(Qs, cost, color)
    plt.show()

#a()
b()
