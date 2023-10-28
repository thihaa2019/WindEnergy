import numpy as np
# demandSimulate.m
def demandSimulate(α, K0, σ, n_step, n_sim, maturity, P0):
    
    dt = maturity/n_step
    priceMatrix = np.zeros((n_sim, n_step+1))
    priceMatrix[:,0] = np.ones(n_sim) * P0
    
    dW = np.random.normal(0,1,size = (n_sim,n_step) ) * np.sqrt(dt)
    
    for i in range(1,n_step+1):
        priceMatrix[:,i] = priceMatrix[:,i-1] + α * (K0 - priceMatrix[:,i-1]) * dt + σ * dW[:,i-1]
        
    return priceMatrix