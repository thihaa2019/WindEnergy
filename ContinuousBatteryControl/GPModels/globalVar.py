import numpy as np
from scipy.stats import qmc
# Batch design parameters
sites = 2**9; batch = 20; nsim = sites * batch
k = 9
# parameters 

    
maturity = 24 * 2 # units are in hours
global dt; dt = 15/60;
nstep = int(maturity/dt)
# B and I range
B_min = -2.5
B_max = 2.5
I_max = 10
# OU params
alpha = 0.5; m0 = 0; sigma = 1

bounds  = sigma**2 *(1- np.exp(-2* alpha *48))/(2*alpha)

# uniform is bad, sample from Xt directly, 

# samplings
sampler = qmc.Sobol(d=2, scramble=False)
# 1024 samples : simple spacde fillling designs
W = sampler.random_base2(m=k)
l_bounds = [-3, 0]
u_bounds = [3, 10]
W= qmc.scale(W, l_bounds, u_bounds)
global X_prev1; X_prev1 = W[:,0]
#print(np.min(X_prev1),np.max(X_prev1))
global I_next1; I_next1 = W[:,1];    # Ic in[0,Imax = 10]

sampler = qmc.Sobol(d=2, scramble=False)
# 1024 samples : simple spacde fillling designs
W = sampler.random_base2(m=8)
l_bounds = [-3, 0]
u_bounds = [3, 2]
W= qmc.scale(W, l_bounds, u_bounds)
global X_prev2; X_prev2= W[:,0]
global I_next2; I_next2 = W[:,1];    # Ic in[0,Imax = 10]
#print(np.min(I_next2),np.max(I_next2))
sampler = qmc.Sobol(d=2, scramble=False)
# 1024 samples : simple spacde fillling designs
W = sampler.random_base2(m=8)
l_bounds = [-3, 8]
u_bounds = [3, 10]
W= qmc.scale(W, l_bounds, u_bounds)
global X_prev3; X_prev3 = W[:,0]
global I_next3; I_next3 = W[:,1];    # Ic in[0,Imax = 10]

X_prev = np.concatenate((X_prev1,X_prev2,X_prev3))
I_next = np.concatenate((I_next1,I_next2,I_next3))
t = np.arange(0,maturity+dt,dt)
