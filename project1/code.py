import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
np.random.seed(79)

def f(x, c):
    if np.any(x <= 0):
        return np.inf
    return (c @ x) - np.sum(np.log(x))

def grad(x, c):
    return (c - 1/x)

def parta():
    m = 100
    n = 500
    data = get_data(m, n)
    x, w, num_steps, f_log, decremant_log = centering_step(*data)
    num_iter = len(f_log)
#    plt.plot(np.arange(num_iter), np.log(decremant_log))
#    plt.show()
    plt.scatter(np.arange(num_iter), np.log(f_log - f_log[-1] + 1e-6))
    plt.plot(np.arange(num_iter), np.log(f_log - f_log[-1] + 1e-6))
    plt.show()

def centering_step(A, b, c, x0, epsilon=1e-6, alpha=0.1, beta=0.7):
    x = x0
    f_log = []
    decremant_log = []
    num_steps = 0
    while True:
        H_inverse_diag = x**2
        gradient = grad(x, c)
        #names follow page 673 of the book
        S = -(A * H_inverse_diag) @ A.T
        b_tilde = A @ (H_inverse_diag * gradient)
        w = np.linalg.solve(S, b_tilde)
        delta_x = -H_inverse_diag * (gradient + A.T @ w)
        decremant = np.sqrt((delta_x * H_inverse_diag) @ delta_x)

        decremant_log.append(decremant**2/2)
        f_log.append(f(x, c))
        if decremant**2/2 <= epsilon:
            break

        num_steps += 1
        t = 1
        dot_prod = gradient @ delta_x
        while f(x + t*delta_x, c) > f(x, c) + alpha * t * dot_prod:
            t *= beta

        x += t * delta_x
    
    f_log = np.array(f_log)
    decremant_log = np.array(decremant_log)
    return x, w, num_steps, f_log, decremant_log

def get_data(m, n):
    A = np.random.rand(m, n)
    x0 = np.abs(np.random.rand(n))
    b = A @ x0
    c = np.random.rand(n)
    return A, b, c, x0

def solve_with_cvx(A, b, c, x0):
    x = cp.Variable(x0.shape)
    obj = cp.Minimize(c @ x - cp.sum(cp.log(x)))
    problem = cp.Problem(obj, [A @ x == b])
    problem.solve()
    return problem.value, x.value

parta()
