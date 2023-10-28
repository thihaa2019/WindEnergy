from demandSimulate import demandSimulate
from globalVar import *
from GPOneStepOptimizer import minimize,one_step_objective,finalCost


import numpy as np
import GPy
from multiprocess import Pool



def runner():

 #create X_{T-1} to X_T


    # E[V(t+1) | X_t-1 = x_(t-1)]
    global Model ; Model = [None] * (nstep)
    gpMdl = None
    ctrl = None
    ctrlModel = [None] * (nstep)
    V_opt_start = None
    B_opt_start = None
    Model[nstep-1] = finalCost

    for iStep in range(nstep,0,-1):

        if iStep!= nstep:
            kernel  = GPy.kern.Matern52(ARD = True, input_dim=2)
            X_train = np.column_stack((X_prev, I_next))
            y_train = costNext
            # E[V_t given X_t-1,I_t]

            gpMdl =  GPy.models.GPRegression(X_train,y_train.reshape(-1,1),kernel)
            gpMdl.optimize(start = V_opt_start)
            V_opt_start = gpMdl.param_array
            print(iStep)
            print(np.mean((y_train-gpMdl.predict(X_train)[0].flatten())**2))
            print("\n")
            Model[iStep-1] = gpMdl
            # assuming we generated, X_t-2,I_t-1
            # create path from X_t-2 to X_t-1, then. we optimize on X_t-1, I_t-1+ Bt dt with V_t to get V_t-1
            demandMatrix = demandSimulate(alpha, m0, sigma, 1, len(X_prev), dt, X_prev);
            optimal_B = np.zeros(len(X_prev))

            ## Parallel Optimize B* as function of(Xt-1,I)
            LB = np.maximum(B_min, (-I_next)/dt)
            UB = np.minimum(B_max, (I_max-I_next)/dt)
            sample_num = len(X_prev)
            args =[(one_step_objective,demandMatrix[i,1],I_next[i],gpMdl,LB[i],UB[i]) for i in range(sample_num)]
            #p = Pool()
            #optimal_B = np.array(p.map(minimize,args))
            idx = 0
            for arg in args:
                optimal_B[idx] = minimize(arg)
                idx+=1

            gp_test = np.column_stack((demandMatrix[:,1], I_next + optimal_B *dt))
            costNext = np.abs((demandMatrix[:,1]+optimal_B)**2) *dt +gpMdl.predict(gp_test)[0].flatten()

        else:

            demandMatrix = demandSimulate(alpha, m0, sigma, 1, len(X_prev), dt, X_prev);
            optimal_B = np.zeros(len(X_prev))

            ## Parallel Optimize B* as function of(Xt-1,I)
            LB = np.maximum(B_min, (-I_next)/dt)
            UB = np.minimum(B_max, (I_max-I_next)/dt)
            sample_num = len(X_prev)
            args =[(one_step_objective,demandMatrix[i,1],I_next[i],finalCost,LB[i],UB[i]) for i in range(sample_num)]
            #p = Pool()
            #optimal_B = np.array(p.map(minimize,args))
            idx = 0
            for arg in args:
                optimal_B[idx] = minimize(arg)
                idx+=1

            gp_test = np.column_stack((demandMatrix[:,1], I_next + optimal_B *dt))
            costNext = np.abs((demandMatrix[:,1]+optimal_B)**2) *dt + finalCost(gp_test[:,1])



        # memorize controls
        kernel2 = GPy.kern.Matern32(ARD= True,input_dim=2)
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