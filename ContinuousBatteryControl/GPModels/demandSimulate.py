import numpy as np

def demandSimulate(α, m, σ, n_step, n_sim, maturity, P0,given_W = False, W = None):
    
    dt = maturity/n_step
    priceMatrix = np.zeros((n_sim, n_step+1))
    priceMatrix[:,0] = np.ones(n_sim) * P0
    
    if not given_W:
        dW = np.random.normal(0,1,size = (n_sim,n_step) ) * np.sqrt(dt)
    
    else:
        dW = W
    
    
    for i in range(1,n_step+1):
        priceMatrix[:,i] = priceMatrix[:,i-1] + α * (m - priceMatrix[:,i-1]) * dt + σ * dW[:,i-1]
        
    return priceMatrix