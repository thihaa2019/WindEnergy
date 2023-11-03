from demandSimulate import demandSimulate
from globalVar import *
from GPOneStepOptimizer import *


import numpy as np
import GPy
from multiprocess import Pool


### Value is RBF Control is RBF
## USE Derivative information
def runner(include_finalcost = True):
    if include_finalcost:
        finalCost = finalCost1
    else:
        finalCost = finalCost2
    

    # create (X_{T-1},I_T)
    X_prev,I_next = create_samples()
    # create conditional expectation models to save 
    global Model ; Model = [None] * (nstep)
    gpMdl = None
    # create control models to save
    ctrl = None
    ctrlModel = [None] * (nstep)
    # init warm-starters for emulators
    V_opt_start = None
    B_opt_start = None
    # this computes E[V(X_T)|I_(T-1)], no regression 
    Model[nstep-1] = finalCost

    # assume we generated X_T-2,I_(T-1)
    #X_prev,I_next = create_samples()
    # create X_(T-2)->X_(T-1)
    demandMatrix = demandSimulate(alpha, m0, sigma, 1, len(X_prev), dt, X_prev);
   
    # finding controls at T-1
    optimal_B = np.zeros(len(X_prev))

    LB = np.maximum(B_min, (-I_next)/dt)
    UB = np.minimum(B_max, (I_max-I_next)/dt)
    sample_num = len(X_prev)
    # not using derivative is better and no difference
    args =[(one_step_objective,None,demandMatrix[i,1],I_next[i],finalCost,LB[i],UB[i]) for i in range(sample_num)]
    #p = Pool()
    #optimal_B = np.array(list(p.imap(minimize,args)))
    #print(optimal_B)

    idx = 0
    for arg in args:
        optimal_B[idx] = minimize(arg)
        idx+=1
    # cost(T-1)|X(T-2) = (B(T-1)+X(T-1)**2)*dt + E[X_T|I_(T-1)]
    gp_test = np.column_stack((demandMatrix[:,1], I_next + optimal_B *dt))
    costNext = np.abs((demandMatrix[:,1]+optimal_B)**2) *dt + finalCost(gp_test[:,1])


    # memorize controls

    # learn B(T-1) as function of (X(T-1),I(T-1))
    kernel2 = GPy.kern.RBF(ARD= True,input_dim=2)
    ctrl = GPy.models.GPRegression(np.column_stack((demandMatrix[:,1], I_next)), optimal_B.reshape(-1,1),kernel2)
    ctrl.optimize(start = B_opt_start)
    B_opt_start = ctrl.param_array
    ctrlModel[nstep-1] = ctrl
    
    for iStep in range(nstep-1,0,-1):
        # do regression to learn  cost(t)|X(t-1) as function of X(t-1) and I(t) where t = T-1,T-2,...,1
        kernel1 = GPy.kern.RBF(input_dim =2, ARD = True)
        X_train = np.column_stack((X_prev, I_next))
        y_train = costNext

        gpMdl =  GPy.models.GPRegression(X_train,y_train.reshape(-1,1),kernel1)
        gpMdl.optimize(start = V_opt_start)
        V_opt_start = gpMdl.param_array
        Model[iStep-1] = gpMdl
        ############################
        #######Print MSE############
        ############################
        print(iStep)
        print("Values:")
        gpMSE = (y_train-gpMdl.predict(X_train)[0].flatten())**2
        print(np.mean(gpMSE))
        print(gpMdl.param_array)
        print("Controls:")
        gpMSE = (optimal_B-ctrl.predict(np.column_stack((demandMatrix[:,1], I_next)))[0].flatten())**2
        print(np.mean(gpMSE))
        print(ctrl.param_array)
        print("\n")
        
        # assuming we generated, X_t-2,I_t-1 
        #X_prev,I_next = create_samples()
        # create path from X_t-2 to X_t-1,
        demandMatrix = demandSimulate(alpha, m0, sigma, 1, len(X_prev), dt, X_prev);
        # finding controls at t-1
        optimal_B = np.zeros(len(X_prev))
        LB = np.maximum(B_min, (-I_next)/dt)
        UB = np.minimum(B_max, (I_max-I_next)/dt)
        sample_num = len(X_prev)
        args =[(one_step_objective,one_step_derivative,demandMatrix[i,1],I_next[i],gpMdl,LB[i],UB[i]) for i in range(sample_num)]
        #p = Pool()
        #optimal_B = np.array(list(p.imap(minimize,args)))
        idx = 0
        for arg in args:
            optimal_B[idx] = minimize(arg)
            idx+=1


        gp_test = np.column_stack((demandMatrix[:,1], I_next + optimal_B *dt))

        # learn cost(t-1)|x(t-2)
        costNext = np.abs((demandMatrix[:,1]+optimal_B)**2) *dt +gpMdl.predict(gp_test)[0].flatten()

        # use GP to learn control (t-1)

        # set prior with previous model parameter as mean
        kernel2 = GPy.kern.RBF(ARD= True,input_dim=2)
        ctrl = GPy.models.GPRegression(np.column_stack((demandMatrix[:,1], I_next)), optimal_B.reshape(-1,1),kernel2)
        ctrl.optimize(start = B_opt_start)
        B_opt_start = ctrl.param_array
        ctrlModel[iStep-1] = ctrl

    return Model,ctrlModel

def optimal_V(I0,X0,nsim,Model,steps):
    Xs = demandSimulate(alpha, m0, sigma, steps, nsim, maturity, X0)
    Bts = np.zeros((nsim,steps))
    Is = np.zeros((nsim,steps+1)); Is[:,0] = I0
    running_cost = np.zeros((nsim,steps))
    for i in range(steps):
        ## Parallel Optimize B* as function of(Xt-1,I)
        LB = np.maximum(B_min, (-Is[:,i])/dt);UB = np.minimum(B_max, (I_max-Is[:,i])/dt)
        sample_num = len(LB)
        args =[(one_step_objective,Xs[idx,i],Is[idx,i],Model[i],LB[idx],UB[idx]) for idx in range(sample_num)]
        idx = 0
        # no parallel
        for arg in args:
            Bts[idx,i] = minimize(arg)
            idx+=1

        Is[:,i+1] = Is[:,i]+Bts[:,i]*dt
        running_cost[:,i] = np.abs((Xs[:,i]+Bts[:,i])**2) *dt 
    total_running_cost = np.sum(running_cost,axis = 1)
    total_cost = np.mean(total_running_cost+200 * np.maximum(5-Is[:,-1],0))
    return total_cost

if __name__ == "__main__":
   Model = runner()
#print(optimal_V(5,0,1000,Model,nstep))