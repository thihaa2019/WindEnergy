import numpy as np

# Batch design parameters
sites = 512; batch = 20; nsim = sites * batch

# parameters

#storage 
global k; k = np.array([1,100000,5]);
global switchCost; switchCost = 10;
global IminMax; IminMax = np.array([0,10]) 
global BminMax; BminMax = np.array([-6,6])

global no_regime; no_regime = 2
global degree; degree = 3

maturity = 24 * 2 # units are in hours

global dt; dt = 15/60;
nstep = int(maturity/dt)