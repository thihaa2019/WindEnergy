# this is V(t,Xt,It) before optimization of Bt
import numpy as np
from globalVar import dt
from scipy import optimize

def finalCost1(I):
    return 200*np.maximum(5-I,0)
def finalCost2(I):
    return 0


def one_step_objective(B,X,I,mdl):
    next_step = np.array([X,I +B[0]*dt])
    if mdl == finalCost1 or mdl==finalCost2:
        objective_func = np.abs((X+B[0])**2) * dt  + mdl(next_step[1] )
    else:
        objective_func = np.abs((X+B[0])**2) * dt + mdl.predict(next_step.reshape(1,-1))[0].flatten()[0]


    return objective_func

# derivative for final cost not implemented
def one_step_derivative(B,X,I,mdl):
    if mdl == finalCost1 or mdl==finalCost2:
        return None

    next_step = np.array([X,I +B[0]*dt])
    objective_der = 2 * (X+B[0]) * dt + mdl.predictive_gradients(next_step.reshape(1,-1))[0][:,1].flatten()[0] * dt
    return objective_der

def minimize(args):
    f,grad_f,x,i,mdl,lb,ub = args
    if mdl == finalCost1 or mdl==finalCost2:
        grad_f = None
    bnds = optimize.Bounds(lb, ub )
    starter = np.maximum(lb,np.minimum(-x,ub))
    res = optimize.minimize(f, jac= grad_f,x0=starter, args=(x,i,mdl),method = "L-BFGS-B", bounds = bnds)
    return res.x[0]

