import numpy as np

N = 50
m = 1500
d = 200
tmp = np.zeros(51)
tmp[10:20]=-10*np.array(range(1,11))[:]
tmp[20:51] = +6*np.array(range(1,32))-100
h = 100 * np.sin(np.array(range(1,N+2))/(N+1)*5* np.pi/2+ np.pi/4) +tmp
eta = .26*35*10**6
rho = 1.2
A = 2.4
c_d = .5
C_D = .5*rho*A*c_d
P = 1500
F = 2  #; % total initial fuel
g = 9.8  #; % acceleration due to gravity

