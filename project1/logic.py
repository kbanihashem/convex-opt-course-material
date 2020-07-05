import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def l2(x):
    return np.linalg.norm(x)

def f(x, c):
    if np.any(x <= 0):
        return np.inf
    return (c @ x) - np.sum(np.log(x))

def grad(x, c):
    return (c - 1/x)

def infeasible_barrier(A, b, c, mu=50, duality_threshold=1e-3, *args, **kwargs):
    m, n = A.shape
    x = np.ones(n)
    log = dict()
    log['center_steps'] = 0
    log['newton_steps'] = 0
    log['log_duality_gap'] = []
    log['cumalative_newton'] = []
    log['history'] = []
    t = 1
    while True:
        x, w, log_center = infeasible_centering_step(A, b, t * c, x, *args, **kwargs)
        log['center_steps'] += 1
        log['newton_steps'] += log_center['num_steps']
        log['log_duality_gap'].append(np.log(m / t))
        log['cumalative_newton'].append(log['newton_steps'])
        log['history'].append((log_center['num_steps'], m / t))
        if m / t < duality_threshold:
            break
        t *= mu

    dual_v = w / t
    dual_lambda = 1 / (t * x)
    return x, log, dual_v, dual_lambda

def infeasible_centering_step(A, b, c, x0, epsilon=1e-8, alpha=0.4, beta=0.7, max_step=50):
    x = x0.copy()
    log = {
            'r_norm': [],
            'r_primal_norm': [],
            'r_dual_norm': [],
            'f_value': [],
            }
    log['num_steps'] = 0
    log['maxed_out_iterations_'] = False
    m, n = A.shape
    v = np.ones(m)

    def r_norm(x, v, gradient):
        if np.any(x <= 0):
            return np.inf
        r = np.hstack([grad(x, c) + A.T @ v, A @ x - b])
        return np.linalg.norm(r)

    while True:
        H_inverse_diag = x**2
        #names follow slide 11-12
        g = grad(x, c)
        h = A @ x - b
        #Some names follow page 673 of the book
        S = -(A * H_inverse_diag) @ A.T
        b_tilde = A @ (H_inverse_diag * g) - h
        w = np.linalg.solve(S, b_tilde)

        delta_x = -H_inverse_diag * (g + A.T @ w)
        delta_v = w - v

        base_r = r_norm(x, v, g)
        if base_r <= epsilon:
            break
        if log['num_steps'] >= max_step:
            log['maxed_out_iterations_'] = True
            break

        log['r_norm'].append(base_r)
        log['r_primal_norm'].append(np.linalg.norm([A @ x - b]))
        log['r_dual_norm'].append(np.linalg.norm([g + A.T @ v]))
        log['f_value'].append(f(x, c))

        log['num_steps'] += 1
        t = 1
        while r_norm(x + t * delta_x, v + t * delta_v, g) > (1 - alpha * t) * base_r:
            t *= beta
        x += t * delta_x
        v += t * delta_v
    
    for name, arr in log.items():
        if name[-1] == '-':
            continue
        log[name] = np.array(arr)

    return x, w, log

def barrier(A, b, c, x0, mu=10, duality_threshold=1e-8, *args, **kwargs):
    m, n = A.shape
    t = 1
    x = x0.copy()
    log = dict()
    log['center_steps'] = 0
    log['newton_steps'] = 0
    log['log_duality_gap'] = []
    log['cumalative_newton'] = []
    log['history'] = []
    while True:
        x, w, log_center = centering_step(A, b, t * c, x, *args, **kwargs)
        log['center_steps'] += 1
        log['newton_steps'] += log_center['num_steps']
        log['log_duality_gap'].append(np.log(m / t))
        log['cumalative_newton'].append(log['newton_steps'])
        log['history'].append((log_center['num_steps'], m / t))
        if m / t < duality_threshold:
            break
        t *= mu
    return x, log

def phase1(A, b, c, Axb_threshold=1e-6):
    m, n = A.shape
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if np.max(np.abs(A @ x - b)) > Axb_threshold:
        return 'infeasible3', np.zeros(n)

    if np.all(x >= 0):
        return 'feasible1', x

    s = np.sum(A, axis=1)
    A_bar = np.hstack([A, -s[:,None]])
    b_bar = b - s
    c_bar = np.zeros(n + 1)
    c_bar[-1] = 1

    t = 2 - np.min(x)
    z = x + (t - 1)
    x_bar = np.hstack([z, t])
    zt, log = barrier(A_bar, b_bar, c_bar, x_bar) 
    t = zt[-1]
    z = zt[:-1]
    x = z + (1 - t)
    if t < 1:
        return 'feasible2', x
    else:
        return 'infeasible4', x

def lp(A, b, c):
    m, n = A.shape
    status, x0 = phase1(A, b, c)
    if status[:-1] == 'infeasible':
        return status, np.zeros(n)
    x_star, log = barrier(A, b, c, x0)
    return status, x_star

def centering_step(A, b, c, x0, epsilon=1e-8, alpha=0.1, beta=0.7, infeasible_method=False):
    x = x0.copy()
    log = {
            'decremant': [],
            'f_value': [],
            }
    log['num_steps'] = 0
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
        log['decremant'].append(decremant_squared/2)
        log['f_value'].append(f(x, c))
        if decremant_squared/2 <= epsilon:
            break

        log['num_steps'] += 1
        t = 1
        dot_prod = gradient @ delta_x
        while f(x + t*delta_x, c) > f(x, c) + alpha * t * dot_prod:
            t *= beta
        x += t * delta_x
    
    for name, arr in log.items():
        log[name] = np.array(arr)

    return x, w, log
