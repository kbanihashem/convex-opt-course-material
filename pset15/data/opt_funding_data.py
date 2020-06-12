import numpy as np
# data for optimal investment to fund an excess stream problem
n  = 6;
T  = 12;
rp = 0.05/12;
rn = 0.09/12;

E  = np.array([1, 1, 8, 5, 1, 2, 2, 8, 6, 0, 8, 1]);                   # expense
C  = np.array([0.0060, 0.0050, 0.0030, 0.0070, 0.0050, 0.0040]); # coupon
P  = np.array([0.9870, 0.9805, 0.9761, 0.9946, 0.9783, 0.9680]); # bond price
M  = np.array([     3,      4,      6,      6,      9,     10]); # time to maturity


A  = np.zeros((T,n)); # A: (time x coupon_type) A=[a_{t,i}]
for i in range(n):
    A[:M[i]-1,i] = C[i];
    A[M[i]-1,i] = C[i]+1;

###############################################################
# computing bond prices
#  use this formula to generate a bond price vector P
#
# Y = [0.125 0.120 0.085 0.095 0.090 0.088]'/12; # bond yield
# P = C.*(1-(1+Y).^-M)./Y+(1+Y).^-M;             # bond price

