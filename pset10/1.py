import numpy as np
import cvxpy as cp

def a():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [cp.norm(cp.vstack([x + 2 * y, x - y])) == 0]
    problem = cp.Problem(obj, constraints)
    print(f"a before: {problem.is_dcp()}")

    constraints = [cp.norm(cp.vstack([x + 2 * y, x - y])) <= 0]
    problem = cp.Problem(obj, constraints)
    print(f"a after: {problem.is_dcp()}")
    problem.solve()

def b():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [cp.square(cp.square(x + y)) <= x - y]
    problem = cp.Problem(obj, constraints)
    print(f"b before and after!: {problem.is_dcp()}")
    problem.solve()

def c():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [x >= 0, y >= 0, 1/x + 1/y <=1]
    problem = cp.Problem(obj, constraints)
    print(f"c before: {problem.is_dcp()}")

    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonneg=True)
    obj = cp.Minimize(0)
    constraints = [cp.inv_pos(x) + cp.inv_pos(y) <=1, x >= 0, y >= 0]
    problem = cp.Problem(obj, constraints)
    print(f"c after: {problem.is_dcp()}")
    problem.solve()

def d():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [
            cp.norm(cp.vstack([cp.maximum(x, 1), cp.maximum(y, 2)])) <= 3 * x + y
            ]
    problem = cp.Problem(obj, constraints)
    print(f"d before: {problem.is_dcp()}")

    x = cp.Variable()
    y = cp.Variable()
    a = cp.Variable()
    b = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [
            cp.norm(cp.vstack([a, b])) <= 3 * x + y,
            a >= cp.maximum(x, 1),
            b >= cp.maximum(y, 2),
            ]
    problem = cp.Problem(obj, constraints)
    print(f"d after: {problem.is_dcp()}")
    problem.solve()

def e():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [
            x * y >= 1,
            x >= 0,
            y >= 0
            ]
    problem = cp.Problem(obj, constraints)
    print(f"e before: {problem.is_dcp()}")

    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [
            cp.inv_pos(y) - x <= 1,
            x >= 0,
            ]
    problem = cp.Problem(obj, constraints)
    print(f"e after: {problem.is_dcp()}")
    problem.solve()

def f():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(x)
    constraints = [
            (x + y)**2 / cp.sqrt(y) <= x - y + 5
            ]
    problem = cp.Problem(obj, constraints)
    print(f"f before: {problem.is_dcp()}")

    x = cp.Variable()
    y = cp.Variable()
    a = cp.Variable()
    b = cp.Variable()
    obj = cp.Minimize(x)
    constraints = [
            cp.quad_over_lin(a, b) <= x - y + 5,
            a == x + y,
            b <= cp.sqrt(y),
            ]
    problem = cp.Problem(obj, constraints)
    print(f"f after: {problem.is_dcp()}")
    problem.solve()

def g():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(x)
    constraints = [
            x**3 + y**3 <= 1,
            ]
    problem = cp.Problem(obj, constraints)
    print(f"g before: {problem.is_dcp()}")
    problem.solve()
    print(f"g solution: {x.value:.2f}")

def h():
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [
            x + z <= 1 + cp.sqrt(x * y - z**2),
            x >= 0,
            y >= 0,
            ]
    problem = cp.Problem(obj, constraints)
    print(f"h before: {problem.is_dcp()}")
   
    t = cp.Variable()
    constraints = [
            x + z - 1 <= t,
            cp.norm(cp.vstack([x, t])) <= cp.geo_mean(cp.vstack([x, y])),
            x >= 0,
            y >= 0,
            ]
    problem = cp.Problem(obj, constraints)
    print(f"h after: {problem.is_dcp()}")
    problem.solve()

a()
b()
c()
d()
e()
f()
g()
h()
