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
        x, w, log = centering_step(*data)
        num_steps, decremant_log, f_log = log
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

def partb():
    m = 100
    n = 500
    for i in range(10):
        data = get_data(m, n, random_state=i)
        A, b, c, x0 = data
        x = barrier(A, b, c, x0)
        cvx_x = solve_b_with_cvx(A, b, c, x0)
        diff = x - cvx_x
        error = np.linalg.norm(diff) / np.linalg.norm(x)

        print(i)
        print(f'error: {error}')

def phase1(A, b, c, Axb_threshold=1e-6):
    m, n = A.shape
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if np.max(np.abs(A @ x - b)) > Axb_threshold:
        print(3)
        return 'infeasible', np.zeros(n)

    if np.all(x >= 0):
        print(1)
        return 'feasible', x

    s = np.sum(A, axis=1)
    A_bar = np.hstack([A, -s[:,None]])
    b_bar = b - s
    c_bar = np.zeros(n + 1)
    c_bar[-1] = 1

    t = 2 - np.min(x)
    z = x + (t - 1)
    x_bar = np.hstack([z, t])
    zt = barrier(A_bar, b_bar, c_bar, x_bar) 
    t = zt[-1]
    z = zt[:-1]
    x = z + (1 - t)
    if t < 1:
        print(2)
        return 'feasible', x
    else:
        print(4)
        return 'infeasible', x

def lp(A, b, c):
    m, n = A.shape
    status, x0 = phase1(A, b, c)
    if status == 'infeasible':
        return status, np.zeros(n)
    x_star = barrier(A, b, c, x0)
    return 'feasible', x_star

def partc():
    m = 100
    n = 500
    #m, n = n, m
    #A, b, c = get_data_c(m, n, random_state=13)
    A, b, c = get_data_c_2(m, n, random_state=13)
    #A, b, c, _ = get_data(m, n)
    status, x_star = lp(A, b, c)
    print(status)
    print(c @ x_star)
    cvx_x = solve_b_with_cvx(A, b, c, c)
    print(c @ cvx_x)
    print(np.max(np.abs(A @ x_star - b)))
    print(np.max(np.abs(A @ cvx_x - b)))


def barrier(A, b, c, x0, mu=10, duality_threshold=1e-8, *args, **kwargs):
    m, n = A.shape
    t = 1
    x = x0.copy()
    while True:
        x, w, log = centering_step(A, b, t * c, x, *args, **kwargs)
        if m / t < duality_threshold:
            break
        t *= mu
    return x


def centering_step(A, b, c, x0, epsilon=1e-8, alpha=0.1, beta=0.7):
    x = x0.copy()
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
    log = num_steps, decremant_log, f_log
    return x, w, log

def get_data_c_2(m, n, random_state=0):
    np.random.seed(random_state)
    A = np.random.randn(m, n)
    A[0] = np.abs(A[0]) 
    x0 = np.abs(np.random.randn(n))
    c = np.random.randn(n)
    A *= 30
    x0 *= 30
    c *= 30

    b = A @ x0
    return A, b, c

def get_data_c(m, n, random_state=0):
    np.random.seed(random_state)

    A = np.random.rand(m, n)
    A[0] = np.abs(A[0]) 
    b = np.random.rand(m)
    c = np.random.rand(n)

    A *= 30
    b *= 30
    c *= 30

    return A, b, c

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

def solve_b_with_cvx(A, b, c, x0):
    x = cp.Variable(x0.shape)
    obj = cp.Minimize(c @ x)
    constraints = [A @ x == b, x >= 0]
    problem = cp.Problem(obj, constraints)
    problem.solve()
    return x.value

def solve_a_with_cvx(A, b, c, x0):
    x = cp.Variable(x0.shape)
    obj = cp.Minimize(c @ x - cp.sum(cp.log(x)))
    constraints = [A @ x == b]
    problem = cp.Problem(obj, constraints)
    problem.solve()
    print(problem.status)
    print(constraints[0].value())
    print(x.value)
    return problem.value, x.value, constraints[0].dual_value

#parta()
partc()
