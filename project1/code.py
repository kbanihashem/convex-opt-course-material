import numpy as np

def a(A, b, c, x0):
    H_inverse_diag = x0**2
    #names follow page 673 of the book
    S = -(A * H_inverse_diag) ) @ A.T
    b_tilde = - A.T @ (H_inverse_diag * g)
    w = np.linalg.sol
