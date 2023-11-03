# this is V(t,Xt,It) before optimization of Bt
import numpy as np
from globalVar import dt
from scipy import optimize



# parallel optimize Bt for each (Xt,It) pair

def minimize(args):
    f,x,i,mdl,lb,ub = args
    bnds = optimize.Bounds(lb, ub )
    res = optimize.minimize(f, x0=(lb+ub)/2, args=(x,i,mdl),method = "L-BFGS-B", bounds = bnds)
    return res.x[0]

def finalCost(I):
    return 200*  np.maximum(5 - I,0)

def one_step_objective(B,X,I,mdl):
    next_step = np.array([X,I +B[0]*dt])
    if mdl ==  finalCost:
        objective_func = np.abs((X+B[0])**2) * dt  + finalCost(next_step[1])
    else:
        objective_func = np.abs((X+B[0])**2) * dt + mdl.predict(next_step.reshape(1,-1))[0].flatten()[0]
    return objective_func