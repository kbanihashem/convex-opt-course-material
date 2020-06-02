import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from data.tens_struct_data import (n, N, A, p,
        g, x_fixed, y_fixed, m, k_tot,
        k_unif, x_unif, y_unif, )

def plot(x_opt, y_opt, k_opt):
    ind_ex = np.where(k_opt < 1e-2); #do not show springs with k < 1e-2
    Aadj = A[:,np.setdiff1d(np.arange(N),ind_ex)];
    Aadj2 = np.dot(Aadj,Aadj.T)-np.diag(np.diag(Aadj@Aadj.T)) != 0;
    
    for i in range(n):
        for j in range(i):
            if Aadj2[i,j]:
                plt.plot([x_opt[i],x_opt[j]],[y_opt[i],y_opt[j]],"bo-")
    plt.plot(x_fixed,y_fixed,'ro');
    plt.show()

x_prime = cp.Variable(n - p)
y_prime = cp.Variable(n - p)
t = cp.Variable()

x = cp.hstack([x_fixed, x_prime])
y = cp.hstack([y_fixed, y_prime])

obj = cp.Minimize(g * (m @ y) + k_tot * t)

constraints = [
        1/2 * ((A.T @ x)**2 + (A.T @ y)**2) <= t
        ]

problem = cp.Problem(obj, constraints)
problem.solve()

k_opt = constraints[0].dual_value
x_opt = x.value
y_opt = y.value
plot(x_opt, y_opt, k_opt)
