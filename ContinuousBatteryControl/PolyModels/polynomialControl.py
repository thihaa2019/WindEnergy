from demandSimulate import demandSimulate
from globalVar import *
from polyOneStepOptimizer import minimize, one_step_objective


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from scipy.stats import qmc
from multiprocess import Pool




def runner():

    #create X_{T-1} to X_T
    X_prev = np.concatenate((X_prev1,X_prev2,X_prev3))
    I_next = np.concatenate((I_next1,I_next2,I_next3))

    #create X_{T-1} to X_T
    demandMatrix = demandSimulate(alpha, m0, sigma, 1, len(X_prev), dt, X_prev);
    #V_T
    finalCost = 200 * np.maximum(5-I_next,0)
    # this gives E[V(t+1) | X_t-1 = x_(t-1)]

    costNext = finalCost;

    global Model ; Model = [None] * (nstep)
    polyMdl = None


    for iStep in range(nstep,0,-1):
        polyMdl = LinearRegression()

        X_train = np.column_stack((X_prev, I_next))
        poly = preprocessing.PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X_train)
        y_train = costNext
        # E[V_t given X_t-1,I_t]
        print(iStep)
        print(costNext[:10])
        print("\n")
        polyMdl.fit(X_poly, y_train)

        # assuming we generated, X_t-2,I_t-1
        # create path from X_t-2 to X_t-1, then. we optimize on X_t-1, I_t-1+ Bt dt with V_t to get V_t-1
        demandMatrix = demandSimulate(alpha, m0, sigma, 1, len(X_prev), dt, X_prev);
        
        optimal_B = np.zeros(len(X_prev))
        
        ## Parallel Optimize B* as function of(Xt-1,I)
        LB = np.maximum(B_min, (-I_next)/dt)
        UB = np.minimum(B_max, (I_max-I_next)/dt)
        sample_num = len(X_prev)
        args =[(one_step_objective,demandMatrix[i,1],I_next[i],polyMdl,LB[i],UB[i]) for i in range(sample_num)]
        p = Pool()
        optimal_B = np.array(p.map(minimize,args))

        
        
        # this gives B*(X(t-1),I(t-1)) = argmin E[(B(t-1)+X(t-1))**2 + V(t+1,XT, I(t-1)+B(t-1)* dt) | x(t-1)]
        poly_test = np.column_stack((demandMatrix[:,1], I_next + optimal_B *dt))
        poly = preprocessing.PolynomialFeatures(degree=3)
        poly_test = poly.fit_transform(poly_test)
        costNext = np.abs((demandMatrix[:,1]+optimal_B)**2) *dt +polyMdl.predict(poly_test)

        Model[iStep-1] = polyMdl
    return Model

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
        # parallel
        p = Pool()
        Bts[:,i] = np.array(p.map(minimize,args))

        Is[:,i+1] = Is[:,i]+Bts[:,i]*dt
        running_cost[:,i] = np.abs((Xs[:,i]+Bts[:,i])**2) *dt 
    total_running_cost = np.sum(running_cost,axis = 1)
    total_cost = np.mean(total_running_cost+200 * np.maximum(5-Is[:,-1],0))
    return total_cost

if __name__ == "__main__":
    Model = runner()
    print(optimal_V(5,0,1000,Model,nstep))
    print(optimal_V(5,0,1000,Model[1:],nstep-1))