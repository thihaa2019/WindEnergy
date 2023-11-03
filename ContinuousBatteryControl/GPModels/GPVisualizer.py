from GPOneStepOptimizer import one_step_derivative,one_step_objective,minimize
from globalVar import *
import numpy as np
from demandSimulate import demandSimulate

# fixed X,I and find B(t)
def B_trajectory(X,I,V_mdls,B_mdls, exact= False,der= False):
    B_star = np.zeros(nstep)

    if der:
        derivative = one_step_derivative
    else:
        derivative = None
    if exact:
        for i in range(nstep):
            LB = np.maximum(B_min, -(I)/dt);UB = np.minimum(B_max, (I_max-I)/dt)
            arg = (one_step_objective,derivative, X,I,V_mdls[i],LB,UB)
            B_star[i] = minimize(arg)
    else:
        inp = np.array([X,I]).reshape(1,-1)
        for i in range(nstep):
            B_star[i] = B_mdls[i].predict(inp)[0].flatten()[0]
    return B_star



def V_trajectory(X0,I0,V_mdls,B_mdls,exact = False,der = False,include_final = False):
    Xs = demandSimulate(alpha, m0, sigma, nstep, 1, maturity, X0)
    Xs = Xs.flatten()
    Bts = np.zeros(nstep)
    Is = np.zeros(nstep+1); Is[0] = I0
    if der:
        derivative = one_step_derivative
    else:
        derivative = None
    running_cost = np.zeros(nstep)
    if exact:
        for i in range(nstep):
            ## Parallel Optimize B* as function of(Xt-1,I)
            LB = np.maximum(B_min, (-Is[i])/dt);UB = np.minimum(B_max, (I_max-Is[i])/dt)
            arg =(one_step_objective,derivative,Xs[i],Is[i],V_mdls[i],LB,UB)
            Bts[i] = minimize(arg)
            Is[i+1] = Is[i]+Bts[i]*dt
            running_cost[i] = np.abs((Xs[i]+Bts[i])**2) *dt 

    else:        
        for i in range(nstep):
            inp = np.array([Xs[i],Is[i]]).reshape(1,-1)
            Bts[i] = B_mdls[i].predict(inp)[0].flatten()[0]
            Is[i+1] = Is[i]+Bts[i]*dt
            running_cost[i] = np.abs((Xs[i]+Bts[i])**2) *dt 
    if include_final:

        final_cost = 200 * np.maximum(5-Is[-1],0)
    else:
        final_cost = 0
    total_cost = np.sum(running_cost)+final_cost

    return total_cost,Xs,Is,Bts,(Xs[:-1]+Bts)
