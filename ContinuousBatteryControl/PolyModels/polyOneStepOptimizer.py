# this is V(t,Xt,It) before optimization of Bt
import numpy as np
from globalVar import dt
from scipy import optimize

from sklearn import preprocessing

# parallel optimize Bt for each (Xt,It) pair

def minimize(args):
    f,x,i,mdl,lb,ub = args
    bnds = optimize.Bounds(lb, ub )
    res = optimize.minimize(f, x0=(lb+ub)/2, args=(x,i,mdl),method = "L-BFGS-B", bounds = bnds)
    return res.x[0]

def one_step_objective(B,X,I,mdl):
    next_step = np.array([X,I +B[0]*dt])
    poly = preprocessing.PolynomialFeatures(degree=3)
    next_step = poly.fit_transform(next_step.reshape(1,-1))
    objective_func = np.abs((X+B[0])**2) * dt + mdl.predict(next_step.reshape(1,-1))
    return objective_func
