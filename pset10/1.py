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

def b():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(0)
    constraints = [cp.square(cp.square(x + y)) <= x - y]
    problem = cp.Problem(obj, constraints)
    print(f"b before and after!: {problem.is_dcp()}")

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
    constraints = [cp.inv_pos(x) + cp.inv_pos(y) <=1]
    problem = cp.Problem(obj, constraints)
    print(f"c after: {problem.is_dcp()}")

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
            cp.inv_pos(y) - x,
            x >= 0,
            ]
    problem = cp.Problem(obj, constraints)
    print(f"e after: {problem.is_dcp()}")

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

def compare_h_with_sol():
    print(f"comparing h with sol")
    #Mine
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    t = cp.Variable()
    obj1 = cp.Minimize(x**2 + y**2 + z**2)
    obj2 = cp.Minimize(x**2/3 + cp.inv_pos(y) - 5 * z + cp.quad_over_lin(x, z) + cp.quad_over_lin(y, z) + cp.quad_over_lin(x, y))
    constraints = [
            x + z - 1 <= t,
            cp.norm(cp.vstack([z, t])) <= cp.geo_mean(cp.vstack([x, y])),
            x >= 0,
            y >= 0,
            ]
    for i, obj in enumerate([obj1, obj2]):
        problem = cp.Problem(obj, constraints)
        problem.solve()
        print(f"problem {i + 1}: my result: status:{problem.status} obj_value:"
                f"{problem.value:.2f}, x: {x.value:.2f}, y: {y.value:.2f}, z: {z.value:.2f}")
    #sol
    t = cp.quad_over_lin(z, y)
    #b = x- t SHOULD work. But it doesn't!!!!!!! I think it's a cvxpy bug
    b = cp.sum(x - t)
    constraints = [
            x + z <= 1 + cp.geo_mean(cp.vstack([y, b])),
            x >= 0,
            y >= 0,
            ]
    for i, obj in enumerate([obj1, obj2]):
        problem = cp.Problem(obj, constraints)
        problem.solve()
        print(f"problem {i + 1}: sol result: status:{problem.status} obj_value:"
                f"{problem.value:.2f}, x: {x.value:.2f}, y: {y.value:.2f}, z: {z.value:.2f}")

a()
b()
c()
d()
e()
f()
g()
h()
compare_h_with_sol()
