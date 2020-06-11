import numpy as np
import cvxpy as cp
from data.opt_funding_data import n, T, rp, rn, E, C, P, M, A

B = cp.Variable(T + 1)
I = cp.Variable(T)
x = cp.Variable(n)

constraints = []
#TODO
