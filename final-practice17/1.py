import numpy as np
import cvxpy as cp
from transform_to_normal_data import n, y0, x
from scipy.stats import norm

D_max = 0.05

y = cp.Variable(n)
ylus_one = y[2:]
ylus_zero = y[1:-1]
ylus_negative_one = y[:-2]

xlus_one = x[2:]
xlus_zero = x[1:-1]
xlus_negative_one = x[:-2]

pos_diff = cp.multiply(ylus_one - ylus_zero, 1/(xlus_one - xlus_zero))
neg_diff = cp.multiply(ylus_zero - ylus_negative_one, 1/(xlus_zero - xlus_negative_one))
R = cp.sum(cp.abs(pos_diff - neg_diff))

upper_bound = np.arange(n) / n + D_max
lower_bound = upper_bound + 1 / n - 2 * D_max
upper_bound = np.minimum(upper_bound, 1)
lower_bound = np.maximum(lower_bound, 0)

upper_bound = norm.ppf(upper_bound)
lower_bound = norm.ppf(lower_bound)
good_indexes = (upper_bound != np.inf) & (lower_bound != -np.inf)

obj = cp.Minimize(R)
epsilon = 0
constraints = [
        y[good_indexes] <= upper_bound[good_indexes],
        y[good_indexes] >= lower_bound[good_indexes],
        y[1:] - y[:-1] >= epsilon,
        ]

problem = cp.Problem(obj, constraints)
problem.solve(verbose=False)
print(problem.value)

