function [cost, stepCost, control, nextInventory, nextRegime, imbalance, batteryOutput] = oneStepOptimization_microgrid(demand, ICord, coeff)
    
    % regime 1 is injection, 2 is withdrawl and 3 is do-nothing.
    nsim = length(demand);
    
    global k; global dt; global IminMax; global degree; global no_regime;
    global switchCost; global BminMax;
    
    B_max = BminMax(2);
    B_min = BminMax(1);

    I_max = IminMax(2);
    I_min = IminMax(1);
    
    maxOutputBattery = ((ICord - I_min)./dt);
    maxInputBattery = - (I_max - ICord)./dt;
    maxOutputBattery = min(B_max,maxOutputBattery);
    maxInputBattery = max(B_min,maxInputBattery);
     
    possibleControl = zeros(nsim,2);
    possibleControl(:,2) = demand.*(demand>0) + abs(maxInputBattery);
    
    %notice that demand has one column, but possible control has 2 columns.
    % demandExControl as a result has 2 columns. 
    demandExControl = demand - possibleControl;
    
    indx1 = demandExControl>maxOutputBattery;
    indx2 = demandExControl<maxInputBattery;
    
    St = indx1.*(demandExControl - maxOutputBattery) + indx2.*(demandExControl - maxInputBattery);
    
    Bt = demandExControl - St;
    
    It = ICord - Bt.*dt;

    costON = k(1)*(possibleControl(:,2).^0.9).*dt + k(2).*St(:,2).*(St(:,2)>0).*dt - k(3).*St(:,2).*(St(:,2)<0).*dt;
    costOFF = k(2).*St(:,1).*(St(:,1)>0).*dt - k(3).*St(:,1).*(St(:,1)<0).*dt; 
    
    espBellmanON = zeros(nsim,no_regime); espBellmanOFF=zeros(nsim,no_regime);  
    nextRegime = 2.*ones(nsim,no_regime); % i initiate this with regime 2
    nextInventory = zeros(nsim,no_regime);     
    stepCost = zeros(nsim,no_regime);    cost = zeros(nsim,no_regime);     
    imbalance = zeros(nsim,no_regime);     batteryOutput = zeros(nsim,no_regime);
    control = zeros(nsim,no_regime);  
    
    %r=1 off and r=2 on.
    for r =1:no_regime
        
        espBellmanOFF(:,r) = costOFF + predict(coeff{1},[demand, It(:,1)]);%polynomialRegressionPredict(demand, It(:,1), degree, coeff{1});

        espBellmanON(:,r) =  costON + predict(coeff{2},[demand, It(:,2)]) + switchCost*(r==1) ;%polynomialRegressionPredict(demand, It(:,2), degree, coeff{2}) 
            
        check = possibleControl(:,2)<0.000001;
        espBellmanON(check,r)=10^11;
        
        indx = espBellmanOFF(:,r) < espBellmanON(:,r);
        nextRegime(:,r) = 1.*indx + 2.*(1-indx);
        stepCost(:,r) =  costOFF.*indx + costON.*(1-indx);
        cost(:,r) = espBellmanOFF(:,r).*indx + espBellmanON(:,r).*(1-indx);
        nextInventory(:,r)= It(:,1).*indx + It(:,2).*(1-indx);
        imbalance(:,r)= St(:,1).*indx + St(:,2).*(1-indx);
        control(:,r) = possibleControl(:,1).*indx + possibleControl(:,2).*(1-indx);
        batteryOutput(:,r) = Bt(:,1).*indx + Bt(:,2).*(1-indx);
        
    end
    
end