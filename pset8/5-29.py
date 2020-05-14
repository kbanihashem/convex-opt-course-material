import numpy as np
coeffs = [1, 18, 118, 324, 238, -360, -400]
roots = np.roots(coeffs) + 3
roots = roots[np.array([0, 3, 4, 5])]
roots = np.real(roots)
roots = np.round(roots, 4)

for root in roots:
    print("\\[")
    a = np.array([-3, 1, 2])
    x = -1/(root + a)
    x_str = ', '.join(map(lambda num: '%.2f' % num, x))
    f_x = (x**2 * a + 2 * x).sum()
    print(f"v={root:.4f} \\implies x(v) = [{x_str}] \\implies f(x) = {f_x:.2f}")
    print("\\]")
