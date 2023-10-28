import time
start_time = time.time()

from global_var import *
from demandSimulate import *
from oneStepOptimization import *
from sklearn import preprocessing


from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


    
def microgrid_gp2d():

    # stock
    alpha = 0.5; K0 = 0; sigma = 2; processType = "Regular";

    # backward simulation
    
    sampler = qmc.Sobol(d=2, scramble=False)
    # 512 samples
    W = sampler.random_base2(m=9)
    # X0 is
    X0 = -10 + 20*W[:,0]; # X0 in [-10,10]
    ICord = 10*W[:,1];    # Ic in[0,Imax = 10]
    ICord_rep = np.repeat(ICord,batch);
    X0_rep = np.repeat(X0,batch);

    demandMatrix = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X0_rep);
    # I0 is 5
    finalCost = 200* np.maximum(5-ICord_rep,0)
    # 20 rows, 512 columns,create 20x512 matrix, filled by column wise with order = "F" (default fill row), 
    # take row sum mean  to get final cost for each starting site
    finalCost = np.mean(finalCost.reshape((batch,sites),order = "F"),axis = 0)
    print("Cost" + str(finalCost[:10]))
    costNext = np.zeros((sites,no_regime));

    for r in range(no_regime):
        costNext[:,r] = finalCost;

    modelIndx = 0; Model = [None] * (nstep)
    gprMdl = [None]*no_regime
    for iStep in range(nstep,0,-1):
        for r in range(no_regime):
            
            kernel =RBF()
            gprMdl[r] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,\
                                                 alpha=1e-1)
            X_train = np.column_stack((X0, ICord))
            # standardize for convergence guarantee
            #scaler = preprocessing.StandardScaler().fit(X_train)
            #X_scaled = scaler.transform(X_train)

            y_train = costNext[:, r]
            gprMdl[r].fit(X_train, y_train)
            #print(iStep,nstep,gprMdl[r].kernel_.get_params())
        # generate X_t 
        demandMatrix = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X0_rep);
        print(costNext[:10,0])
        # optimize basedd on X_t-1,I_t
        cost, _, control, nextInventory, nextRegime, imbalance, batteryOutput =oneStepOptimization_microgrid(
            demandMatrix[:, 1], ICord_rep, gprMdl)

        for r in range(no_regime):
            costNext[:, r] = np.mean(cost[:, r].reshape((batch, sites),order = "F"), axis=0)
    
        Model[modelIndx] = gprMdl.copy();
        modelIndx+=1;    


    # forward simulations 
    np.random.seed(10)
    simOutSample = 200000;

    X0 = np.zeros(simOutSample);
    I0 = 5;
    #X0 is not statationary distribution but started at mean? GP assume stationrity?
    demandMatrix = demandSimulate(alpha, K0, sigma, nstep, simOutSample, maturity, X0);

    inventoryForward = np.zeros((simOutSample,nstep+1));
    inventoryForward[:, 0] = I0

    regimeForward = np.zeros((simOutSample, nstep + 1))
    regimeForward[:, 0] = 0
    
    Bt = np.zeros((simOutSample, nstep))
    St = np.zeros((simOutSample, nstep))
    dieselPower = np.zeros((simOutSample, nstep))
    trueCost = np.zeros((simOutSample, nstep))
    artificialCost = np.zeros((simOutSample, nstep))
    
    costForward = np.zeros((simOutSample, nstep + 1))
    
    for iStep in range(nstep):
        _, _, control, nextInventory, nextRegime, imbalance, batteryOutput = \
        oneStepOptimization_microgrid(demandMatrix[:, iStep], inventoryForward[:, iStep], Model[nstep - iStep-1])
        row_idx = tuple(np.arange(0,simOutSample))
        col_idx = tuple(np.int64(regimeForward[:,iStep]))
        inventoryForward[:, iStep + 1] = nextInventory[row_idx,col_idx]
        regimeForward[:, iStep + 1] = nextRegime[row_idx,col_idx]
        Bt[:, iStep] = batteryOutput[row_idx,col_idx]
        St[:, iStep] = imbalance[row_idx,col_idx]
        dieselPower[:, iStep] = control[row_idx,col_idx]
        
        trueCost[:, iStep] = k[0] * (dieselPower[:, iStep] ** 0.9) * dt + switchCost * (regimeForward[:, iStep + 1] > regimeForward[:, iStep])
        artificialCost[:, iStep] = k[1] * St[:, iStep] * (St[:, iStep] > 0) * dt - k[2] * St[:, iStep] * (St[:, iStep] < 0) * dt
    
    penalty = -200 * (inventoryForward[:, -1] - I0) * (inventoryForward[:, -1] < I0)
    
    costForward = trueCost + artificialCost
    pathWiseCost = np.sum(costForward, axis=1) + penalty
    totalCost = np.mean(pathWiseCost)
    stdDeviation = np.std(pathWiseCost) / np.sqrt(simOutSample)
    totalTrueCost = np.mean(np.sum(trueCost, axis=1))
    totalArtificialCost = np.mean(np.sum(artificialCost, axis=1))
    
    print(f'nsim: {sites} x {batch}, regime=1, totalCost= {totalCost}, totalTrueCost= {totalTrueCost}, totalArtificialCost= {totalArtificialCost}')



if __name__ == "__main__":
    microgrid_gp2d()
    print(f'Running time: {time.time()- start_time} s')