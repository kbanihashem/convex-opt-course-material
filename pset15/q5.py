import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data.opt_funding_data import n, T, rp, rn, E, C, P, M, A
np.set_printoptions(precision=4, suppress=True)

B = cp.Variable(T + 1)
I = cp.Variable(T + 1)
x = cp.Variable(n)
E = np.hstack([0, E])

constraints = [
        B[-1] + I[-1] - E[-1] == 0,
        B[:-1] >= B[1:] * (1/(1 + rn)) + cp.pos(B[1:]) * (1/(1 + rp) - 1/(1 + rn)) - I[:-1] + E[:-1],
        B[0] >= 0,
        I[0] == 0,
        I[1:] == A @ x,
        x >= 0,
        ]

obj = cp.Minimize(x @ P + B[0])
problem = cp.Problem(obj, constraints)
problem.solve()
print("value: ", problem.value)
print(f"optimal x: {x.value}")
print(f"optimal B0: {B[0].value:.4f}")

def plot_balance(x_val):
    B = cp.Variable(T + 1)
    I = cp.Variable(T + 1)
    x = cp.Variable(n)
    constraints = [
            B[-1] + I[-1] - E[-1] == 0,
            B[:-1] >= B[1:] * (1/(1 + rn)) + cp.pos(B[1:]) * (1/(1 + rp) - 1/(1 + rn)) - I[:-1] + E[:-1],
            B[0] >= 0,
            I[0] == 0,
            I[1:] == A @ x,
            x == x_val,
            ]

    obj = cp.Minimize(x @ P + B[0])
    problem = cp.Problem(obj, constraints)
    problem.solve()
    
    plt.scatter(np.arange(T + 1), B.value, color='blue')
    plt.plot(np.arange(T + 1), B.value, color='red')
    plt.show()

print("Optimal: ")
plot_balance(x.value)
print("no bonds: ")
plot_balance(np.zeros(n))
