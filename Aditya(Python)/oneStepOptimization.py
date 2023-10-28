# oneStepOptimization_microgrid.m 
from global_var import *
from sklearn import preprocessing
import numpy as np


def oneStepOptimization_microgrid(demand,ICord,coeff):
    
    # regime 0= injection, 1 = withdrawal, 2  = do nothing
    # Batch design parameters
    nsim = len(demand)


    B_max = BminMax[1];
    B_min = BminMax[0];

    I_max = IminMax[1];
    I_min = IminMax[0];
    
    maxOutputBattery = ((ICord - I_min)/dt); 
    maxInputBattery = - (I_max - ICord)/dt;
    maxOutputBattery = np.minimum(B_max*np.ones(len(maxOutputBattery)),maxOutputBattery);
    maxInputBattery = np.maximum(B_min*np.ones(len(maxInputBattery)),maxInputBattery);
    
    maxOutput2D = np.column_stack((maxOutputBattery,maxOutputBattery))
    maxInput2D = np.column_stack((maxInputBattery,maxInputBattery))
    
    #c_tk
    possibleControl = np.zeros((nsim,2));
    possibleControl[:,1] = demand*(demand>0) + np.abs(maxInputBattery);
    
    #notice that demand has one column, but possible control has 2 columns.
    # demandExControl as a result has 2 columns. 

    # Xt-ct
    demand2D = np.column_stack((demand,demand))
    demandExControl = demand2D - possibleControl;
    
    indx1 = demandExControl>maxOutput2D;
    indx2 = demandExControl<maxInput2D;
    
    St = indx1 * (demandExControl - maxOutput2D) + indx2 * (demandExControl - maxInput2D);
    
    Bt = demandExControl - St;
    
    ICord2D = np.column_stack((ICord,ICord))
    It = ICord2D - Bt * dt;

    costON = k[0] * (possibleControl[:,1]**0.9) * dt + k[1] * St[:,1]*(St[:,1]>0) * dt - k[2] * St[:,1]*(St[:,1]<0) * dt;
    costOFF = k[1] * St[:,0] * (St[:,0]>0)*dt - k[2]*St[:,0]*(St[:,0]<0)*dt; 
    
    espBellmanON = np.zeros((nsim,no_regime)); espBellmanOFF=np.zeros((nsim,no_regime));  
    nextRegime = 1 * np.ones((nsim,no_regime)); # i initiate this with regime 1
    nextInventory = np.zeros((nsim,no_regime));     
    stepCost = np.zeros((nsim,no_regime));    cost = np.zeros((nsim,no_regime));     
    imbalance = np.zeros((nsim,no_regime));     batteryOutput = np.zeros((nsim,no_regime));
    control = np.zeros((nsim,no_regime));   
    
    # r=0 off and r=1 on.
    for r in range(no_regime):
        
        X = np.column_stack((demand, It[:,0]))
        #standardize X
        #scaler = preprocessing.StandardScaler().fit(X)
        #X_scaled = scaler.transform(X)

        pred0 = coeff[0].predict(X)
        espBellmanOFF[:,r] = costOFF +pred0;
        
        X = np.column_stack((demand, It[:,1]))
        #standardize X
        #scaler = preprocessing.StandardScaler().fit(X)
        #X_scaled = scaler.transform(X)

        pred1 = coeff[1].predict(X)
        espBellmanON[:,r] =  costON + pred1 + switchCost*(r==0) ;
        
        check = possibleControl[:,1]<0.000001;
        espBellmanON[check,r]=10**11;
        
        indx = espBellmanOFF[:,r] < espBellmanON[:,r];
        nextRegime[:,r] = 0 *indx + 1 *(1-indx);
        stepCost[:,r] =  costOFF * indx + costON * (1-indx);
        cost[:,r] = espBellmanOFF[:,r] * indx + espBellmanON[:,r] * (1-indx);
        nextInventory[:,r]= It[:,0] * indx + It[:,1] * (1-indx);
        imbalance[:,r]= St[:,0] * indx + St[:,1] * (1-indx);
        control[:,r] = possibleControl[:,0] * indx + possibleControl[:,1] * (1-indx);
        batteryOutput[:,r] = Bt[:,0] * indx + Bt[:,1] * (1-indx); 
        
    return [cost, stepCost, control, nextInventory, nextRegime, imbalance, batteryOutput]