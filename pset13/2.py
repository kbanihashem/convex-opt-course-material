import numpy as np
import cvxpy as cp

def main():
    np.set_printoptions(precision=2, suppress=True)
    b = np.array([400, 80, 400, 200, 400, 400, 80, 400, 100, 500])
    v = np.array([500, 100, 500, 200, 700, 300, 120, 300, 150, 600])
    n = 10
    L = 4
    rho_l = 0.2
    rho_s = 0.3
    rates = (v - b) / v
    C = 2300

    s = cp.Variable(n)
    g = cp.multiply(s, rates)
    N_l = cp.sum(g[:L])
    N_s = cp.sum(g[L:])
    constraints = [
            s >= 0,
            s <= v,
            cp.sum(s) == C,
            N_l <= 0,
            N_s <= 0,
            ]
    obj = cp.Minimize(0)
    problem = cp.Problem(obj, constraints)
    problem.solve()
    if problem.status != 'infeasible':
        print(f"optimal value: {problem.value}")
        print(f"optimal s: {s.value}")
        return
    li = []
    #i = 0: N_l <= 0
    for i in range(2):
        constraints = [
                s >= 0,
                s <= v,
                cp.sum(s) == C,
                (N_l if i == 0 else N_s) <= 0,
                ]
        rho = rho_s if i == 0 else rho_l
        obj = cp.Minimize(rho * cp.pos(N_s + N_l))
        problem = cp.Problem(obj, constraints)
        problem.solve()
        if problem.status == 'optimal':
            li.append((problem.status, problem.value, s.value, i))

    i = 2
    constraints = [
            s >= 0,
            s <= v,
            cp.sum(s) == C,
            N_l >= 0,
            N_s >= 0,
            ]
    obj = cp.Minimize(rho_s * N_s + rho_l * N_l)
    problem = cp.Problem(obj, constraints)
    problem.solve()
    if problem.status == 'optimal':
        li.append((problem.status, problem.value, s.value, i))
    li.sort(key=lambda x: x[1])
    best_value = li[0][1]
    best_s = li[0][2]
    print(f"best value: {best_value:.4f}")
    print(f"best s: {best_s}")
#    print(f"N_s = {N_s.value:.4f}, N_l={N_l.value:.4f}")
#    print(li[0][1], li[0][-1])
#    print(li[1][1], li[1][-1])
#    print(li[2][1], li[2][-1])
    print("evaluating dumb s")
    dumb_s = C / np.sum(v) * v
    g = dumb_s * rates
    n_l = np.sum(g[:L])
    n_s = np.sum(g[L:])
    print(f"n_s: {n_s:.4f}, n_l: {n_l:.4f}")
    #we know it's positive
    print(f"Dumb cost = {(rho_s * n_s + rho_l * n_l):.4f}")

main()
