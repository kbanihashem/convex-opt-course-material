import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from functools import reduce
EPSILON = 1e-8
def verify(condition, text=" ", indentaion=0):
    if np.all(condition):
        print(f"{' ' * 8 * indentaion}condition {text} verified")
    else:
        print(f"{' ' * 8 * indentaion}condition {text} violated")

def verify_zero(value, *args, **kwargs):
    verify(np.abs(value) <= 0 + EPSILON, *args, **kwargs)

#solving original problem
perturb = (0, 0)
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

#just checking everthing is fine
assert problem.status == 'optimal'

np.set_printoptions(precision=3)
print(f"Optimal value: {problem.value:.4f}")
print(f"Optimal x: {x.value}")
print(f"Optimal lambda: {constraints[0].dual_value}")
print("Verifying KKT")
print("Going by the order in page 244 of the book")
x_bar = x.value
lambda_bar = constraints[0].dual_value
verify(np.dot(A, x_bar) - b <= 0 + EPSILON, "f_i(x_bar) <= 0")
verify(lambda_bar >= EPSILON, "lambda >= 0")
verify_zero((np.dot(A, x_bar) - b) * lambda_bar, "lambda_bar * f_i(x_bar) = 0")
verify_zero(2 * P.dot(x_bar) + q + lambda_bar.dot(A), "Gradient condition")

original_value = problem.value
original_lambda_star = constraints[0].dual_value
li = [0, -0.1, 0.1]
possible_deltas = [(x, y) for x in li for y in li]
for perturb in possible_deltas:
    #calculatin pred
    pred = original_value - original_lambda_star[:2].dot(np.array([*perturb]))
    print(f"perturb = {perturb}")
    print(f"\tp_pred: {pred:.4f}")
    #exact
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
        print(f"\tcould'nt solve! status={problem.status}")
    else:
        print(f"\tp_exact = {problem.value:.4f}")
    verify(pred <= problem.value + EPSILON, "pred < exact", indentaion=1)


