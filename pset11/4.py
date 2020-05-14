import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from functools import reduce
from operator import add

#ci is short for convert index
def convert_index(*args):
    return reduce(lambda x, y: x * 2 + y, args)

all_tuples = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
def get_prob(p=None, i=None, j=None, k=None, l=None):
    def is_fine(i0, j0, k0, l0):
        a = [i, j, k, l]
        b = [i0, j0, k0, l0]
        return all(i == ip or i is None for i, ip in zip(a, b))
    return reduce(add, [p[convert_index(i, j, k, l)] for i, j, k, l in all_tuples if is_fine(i, j, k, l)])

p = cp.Variable(16)
constraints = [
        get_prob(p) == 1,
        get_prob(p, i=1) == 0.9,
        get_prob(p, j=1) == 0.9,
        get_prob(p, k=1) == 0.1,
        get_prob(p, i=1, l=0, k=1) == 0.7 * get_prob(p, k=1),
        get_prob(p, l=1, j=1, k=0) == 0.6 * get_prob(p, j=1, k=0),
        p >= 0,
        ]
obj1 = cp.Minimize(get_prob(p, l=1))
obj2 = cp.Maximize(get_prob(p, l=1))
p1 = cp.Problem(obj1, constraints)
p2 = cp.Problem(obj2, constraints)
p1.solve()
p2.solve()

print(f"minimum X_4: {p1.value:.2f}")
print(f"maximum X_4: {p2.value:.2f}")
