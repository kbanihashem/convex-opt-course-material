import numpy as np
import cvxpy as cp
from data.disks_data import n, k, lim, Cgiven, Rgiven, Gindexes, plot_disks

#fixing data
Cgiven = np.asarray(Cgiven)
Rgiven = np.asarray(Rgiven)
G = np.rint(np.asarray(Gindexes)).astype(int)

r = cp.Variable(n)
c = cp.Variable((n, 2))
constraints = [
        c[:k] == Cgiven,
        r[:k] == Rgiven,
        r >= 0,
        ]
for i, j in G:
    constraints.append(cp.norm(c[i] - c[j]) <= r[i] + r[j])

a_and_b = {
        'Perimeter': cp.Minimize(2 * np.pi * cp.sum(r)),
        'Area': cp.Minimize(np.pi * cp.sum_squares(r)),
        }

print(G)
for name, obj in a_and_b.items():
    print(f"Problem name: {name}")
    problem = cp.Problem(obj, constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")
    print(r.value)
    print(c.value)
    plot_disks(c.value, r.value, G, name)
