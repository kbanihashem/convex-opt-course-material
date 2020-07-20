import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data.deconvolution_data import Y, A, B, n, m, d
y = np.reshape(Y, (m,))

def show_image(z):
    plt.imshow(np.reshape(z, (d,d)).T, "gray", interpolation="nearest")
    plt.show()


x = cp.Variable(n)
constraints = [
        y == A @ B @ x
        ]
pics = {}
for norm in [1, 2, 'inf']:
    obj = cp.Minimize(cp.norm(x, norm))
    problem = cp.Problem(obj, constraints)
    problem.solve()
    print('norm: ', norm)
    print('status: ', problem.status)
    print('value: ', problem.value)
    z = B @ x.value
    pics[norm] = z
for norm in pics:
    show_image(pics[norm])
show_image(y)
