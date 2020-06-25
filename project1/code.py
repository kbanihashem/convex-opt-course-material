import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
SHOULD_PLOT_A = True
np.set_printoptions(precision=6, suppress=True)

def f(x, c):
    if np.any(x <= 0):
        return np.inf
    return (c @ x) - np.sum(np.log(x))

def grad(x, c):
    return (c - 1/x)

def parta():
    m = 100
    n = 500
    for i in range(35, 36):
        data = get_data(m, n, random_state=i)
        A, b, c, x0 = data
        x, w, num_steps, f_log, decremant_log = centering_step(*data)
        num_iter = len(f_log)
        decremant_log = f_log - f_log[-1] + 1e-8
        if SHOULD_PLOT_A:
            plt.plot(np.arange(num_iter), np.log(decremant_log))
            plt.scatter(np.arange(num_iter), np.log(decremant_log))
            plt.show()

        primal_slackness = np.max(np.abs(A @ x - b))
        dual_slackness = np.max(np.abs(grad(x, c) + A.T @ w))
        print(i)
        print(f'primal_slackness: {primal_slackness:.6f}')
        print(f'dual_slackness: {dual_slackness:.6f}')
        print(f'num_steps: {num_steps}')


def centering_step(A, b, c, x0, epsilon=1e-10, alpha=0.1, beta=0.7):
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
        decremant_squared = -delta_x @ gradient
        #decremant_squared = (delta_x / H_inverse_diag) @ delta_x

        decremant_log.append(decremant_squared/2)
        f_log.append(f(x, c))
        if decremant_squared/2 <= epsilon:
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

def get_data(m, n, random_state=0):
    np.random.seed(random_state)
    A = np.random.rand(m, n)
    A[0] = np.abs(A[0]) 
    x0 = np.abs(np.random.rand(n))
    c = np.random.rand(n)
    A *= 30
    x0 *= 30
    c *= 30

    b = A @ x0
    return A, b, c, x0

def solve_with_cvx(A, b, c, x0):
    x = cp.Variable(x0.shape)
    obj = cp.Minimize(c @ x - cp.sum(cp.log(x)))
    constraints = [A @ x == b]
    problem = cp.Problem(obj, constraints)
    problem.solve()
    print(problem.status)
    print(constraints[0].value())
    print(x.value)
    return problem.value, x.value, constraints[0].dual_value

parta()
