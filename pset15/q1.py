import numpy as np
import cvxpy as cp
from data.ls_perm_meas_data import m, k, n, A, y, x_true, P
K_P = P
del P
x_true = np.reshape(x_true, n)
y = np.reshape(y, m)
np.set_printoptions(suppress=True)

def get_error(y, y_hat):
    return np.sum((y - y_hat)**2)

before = []
after = []

bad_indexes = set()
max_iter = 10
for iteration in range(max_iter):
    mask = np.array([i not in bad_indexes for i in range(m)])
    x = cp.Variable(n)
    y_hat = A @ x
    error = (y_hat - y)**2
    obj = cp.Minimize(cp.sum(error[mask]))
    problem = cp.Problem(obj, [])
    problem.solve()
    #x is uniform the first time!
    if iteration == 0:
        x_unif = x.value
    bad_indexes = np.argsort(-error.value)[:k]
    bad_indexes_set = set(bad_indexes)

y_hat_bad_indexs_sorted_index = np.argsort(y_hat.value[bad_indexes])
y_true_bad_indexs_sorted_index = np.argsort(y[bad_indexes])
permuation = np.arange(m)
for i in range(k):
    permuation[bad_indexes[y_true_bad_indexs_sorted_index[i]]] = bad_indexes[y_hat_bad_indexs_sorted_index[i]]

P = np.zeros((m, m))
P[np.arange(m), permuation] = 1

x = cp.Variable(n)
y_hat = A @ x
error = (P @ y_hat - y)**2
obj = cp.Minimize(cp.sum(error))
problem = cp.Problem(obj, [])
problem.solve()

x_smart = x.value

print(f"||x_unif - x_true||_2 = {np.linalg.norm(x_unif - x_true)}")
print(f"||x_smart - x_true||_2 = {np.linalg.norm(x_smart - x_true)}")
print(f"our estimate of permutated indexes = {sorted(bad_indexes)}")
real_bad = [i for i in range(m) if abs(P[i,i] - 1) > 1/10]
print(f"real permutated indexes = {real_bad}")
correct_count = len(set(real_bad).intersection(bad_indexes_set))
print(f"got {correct_count}/{k} correct")
