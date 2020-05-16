import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(0)
(m, n) = (300, 100)

A = np.random.rand(m, n)
b = A.dot(np.ones(n))/2
c = -np.random.rand(n)
x = cp.Variable(n)
relaxed_obj = cp.Minimize(c @ x)
constraints = [
        A @ x <= b,
        x >= 0,
        x <= 1,
        ]
problem = cp.Problem(relaxed_obj, constraints)
problem.solve()
lower_bound = problem.value
print(f"relaxed objective: {lower_bound:.4f}")

t_count = 101
t_values = np.linspace(0, 1, t_count)
obj_value = np.zeros(t_count)
maximum_violation = np.zeros(t_count)

for i, t in enumerate(t_values):
    new_x = (x.value > t).astype(int)
    obj_value[i] = np.dot(c, new_x)
    maximum_violation[i] = np.max(np.matmul(A, new_x) - b)

good_indexes = np.arange(t_count)[maximum_violation <= 0]
best_t_index = good_indexes[np.argmin(obj_value[good_indexes])]
print(f"best t_index: {best_t_index}, t: {t_values[best_t_index]}")
upper_bound = obj_value[best_t_index]
U_minus_L = upper_bound - lower_bound
frac = lower_bound / upper_bound
print(f"Upper bound: {upper_bound:.4f}, lower bound: {lower_bound:.4f}, U - L: {U_minus_L:.4f}, frac: {(frac * 100):.2f} %")

feasible = maximum_violation <= 0
plt.plot(t_values[feasible], obj_value[feasible], color='green')
plt.plot(t_values[~feasible], obj_value[~feasible], color='red')
plt.show()
plt.plot(t_values[feasible], maximum_violation[feasible], color='green')
plt.plot(t_values[~feasible], maximum_violation[~feasible], color='red')
plt.show()
