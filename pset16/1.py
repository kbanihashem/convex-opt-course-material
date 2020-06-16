import numpy as np
import cvxpy as cp
from data.quad_metric_data import n, N, N_test, X, Y, X_test, Y_test, d, d_test

P = cp.Variable((n, n), PSD=True)
X_minus_Y = X - Y
#d2 = (X_minus_Y.T @ (P @ X_minus_Y))[np.diag_indices(N)]
d2 = []
for i in range(N):
    d2.append(cp.quad_form(X_minus_Y[:,i], P))
d2 = cp.hstack(d2)
obj = cp.Minimize((cp.sum(d**2 + d2) - 2 * d @ cp.sqrt(d2)) / N)
problem = cp.Problem(obj, [])
problem.solve()
print(f'Problem status: {problem.status}')
print(f'Optimal distance: {problem.value}')
our_P = P.value
X_minus_Y_test = X_test - Y_test
d_test_hat = np.sqrt(((X_minus_Y_test.T).dot(our_P).dot(X_minus_Y_test))[np.diag_indices(N_test)])
test_error = np.mean((d_test_hat - d_test)**2)
print(f'test_error: {test_error}')
