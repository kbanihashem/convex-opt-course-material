import numpy as np
import cvxpy as cp

def get_random_array(shape, method, positive=False):
    if method == 'normal':
        answer = np.random.rand(*shape) * 2 - 1
    else:
        answer = np.random.randn(*shape)
    if positive:
        answer = np.abs(answer)
    return answer

def get_data(m, n, method='normal', random_state=0, part='a'):
    np.random.seed(random_state)
    A = get_random_array((m, n), method)
    A[0] = np.abs(A[0])
    c = get_random_array((n,), method)
    if part == 'a':
        x = get_random_array((n,), method, positive=True)
        b = A @ x
    if part == 'c':
        b = get_random_array((m,), method)
        x = np.zeros(n) 
    return A, b, c, x

def solve_with_cvx(A, b, c, with_log=False):
    x = cp.Variable(c.shape)
    obj = cp.Minimize(
            c @ x - cp.sum(cp.log(x)) else c @ x
            )
    constraints = [A @ x == b, x >= 0]
    problem = cp.Problem(obj, constraints)
    problem.solve()
    return {
            'x_value': x.value,
            'dual_value': constraints[0].dual_value,
            'obj_value': problem.value
            }
