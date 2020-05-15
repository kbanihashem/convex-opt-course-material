import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from functools import reduce

def main():
    a()
    b()

def solve_qp(perturb):
    u1, u2 = -2 + perturb[0], -3 + perturb[1]
    P = np.array([[1, -1/2], [-1/2, 2]])
    q = np.array([-1, 0])
    A = np.array([[1, 2], [1, -4], [5, 76]])
    b = np.array([u1, u2, 1])
    x = cp.Variable(2)

    obj = cp.Minimize(cp.quad_form(x, P) + q @ x)
    constraints = [
            A @ x <= b,
            ]
    problem = cp.Problem(obj, constraints)
    problem.solve()

    if problem.status != 'optimal':
        print(f"Can't solve, status = {problem.status}")
        return

    np.set_printoptions(precision=3)
    print(f"Optimal value: {problem.value:.4f}")
    print(f"Optimal x: {x.value}")
    print(f"Optimal lambda: {constraints[0].dual_value}")

def a():
    pass

def b():
    pass

main()
