
function priceMatrix = demandSimulate(alpha, K0, sigma, nstep, nsim, maturity, P0)
    
    dt = maturity/nstep;
    priceMatrix = zeros(nsim,nstep+1);
    priceMatrix(:,1)= P0;

    dW = normrnd(0,1, nsim, nstep)*sqrt(dt);
    for iStep=2:nstep+1
        priceMatrix(:,iStep) = priceMatrix(:,iStep-1) + alpha*(K0 - priceMatrix(:,iStep-1))*dt + sigma*dW(:,iStep-1);
    end
    
end

