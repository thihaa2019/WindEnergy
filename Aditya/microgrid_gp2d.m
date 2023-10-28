function [totalCost,totalTrueCost,totalArtificialCost] = microgrid_gp2d()
clear;
warning('off','all')

%% Batch design parameters
sites = 500;
batch = 20;
nsim = sites*batch;

%% parameters

% storage 
global k; k = [1, 100000, 5];
global switchCost; switchCost = 10;
global IminMax; IminMax = [0,10];
global BminMax; BminMax = [-6,6];

global no_regime; no_regime = 2;




maturity  = 24*2; % in hours
global dt; dt = 15/60;
nstep = fix(maturity/dt); % 

% stock
alpha = 0.5; K0 = 0; sigma = 2; processType = 'Regular';


%% backward simulation

rng('shuffle')
% load('InventoryDataForProbability.mat');
% load('demandDataForProbability.mat');

% lhs = lhsdesign(0.4*sites,2);
% X0 = -6 + 12*lhs(:,1);
% ICord = 10*lhs(:,2);
% W = [datasample([XP(:,50) XI(:,50)],0.6*sites);[X0,ICord]];
% % W = [[datasample([XP(:,50) XI(:,50)],0.6*sites),datasample(XI(:,50),0.6*sites)];[X0,ICord]];
% % W = [[prctile(XP(:,end),linspace(0.1,99,0.6*sites))' , prctile(XI(:,end),linspace(0.1,99,0.6*sites))'];[P0,ICord]];
% X0 = W(:,1);
% ICord = W(:,2);

p = sobolset(2);
W = net(p,floor(sites));
X0 = -10 + 20*W(:,1);
ICord = 10*W(:,2);

% lhs = lhsdesign(sites,2);
% X0 = -6 + 12*lhs(:,1);
% ICord = 10*lhs(:,2);


ICord_rep = repelem(ICord,batch);
X0_rep = repelem(X0,batch);

demandMatrix = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X0_rep);

finalCost = -200*((ICord_rep-5)).*(ICord_rep<5); %2*0*demandMatrix(:,end).*(ICord_rep-5).*(ICord_rep<5);
finalCost = mean(reshape(finalCost,[batch,sites]),1)';
disp(finalCost)
costNext = zeros(sites,no_regime);

for r=1:no_regime
    costNext(:,r) = finalCost;
end
disp(costNext(:,1))

modelIndx = 1;

tic
for iStep = nstep:-1:1

    for r =1:no_regime
        gprMdl{r} = compact(fitrgp([X0, ICord],costNext(:,r), 'KernelFunction','ardsquaredexponential'));
    end 

%     lhs = lhsdesign(0.4*sites,2);
%     X0 = -6 + 12*lhs(:,1);
%     ICord = 10*lhs(:,2);
%     W = [datasample([XP(:,50) XI(:,50)],0.6*sites);[X0,ICord]];
%     X0 = W(:,1); X0_rep = repelem(X0,batch);    
%     ICord = W(:,2);     ICord_rep = repelem(ICord,batch);  
    
    demandMatrix = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X0_rep);
    
    [cost, ~, control, nextInventory, nextRegime, imbalance, batteryOutput] = oneStepOptimization_microgrid(demandMatrix(:,2), ICord_rep, gprMdl);

    disp(iStep)
    for r=1:no_regime
        costNext(:,r) = mean(reshape(cost(:,r),[batch,sites]),1)';
    end
    disp(costNext(1:10,1))
    contains_negative = any(costNext(:,1)<0);
    disp(contains_negative)
    Model{modelIndx} = gprMdl;
    modelIndx=modelIndx+1;    

%     if iStep==1
%         asset= -7+14*rand(10000,1);
%         inventory= 10*rand(10000,1);
%         [costNext, ~, control, nextInventory, nextRegime, imbalance, batteryOutput] = oneStepOptimization_microgrid(asset , inventory, Model{1});
% %         
% %         [valueNext, nextInventory, profit, nextRegime] = oneStepOptimization(asset*1000, inventory , gprMdl);        
% %         control = nextInventory - inventory;
% %         
% %         control(control>1)=1;
% %         control(control<-1)=-1;
%         figure(1)
%         scatter(asset, inventory, 2, control(:,1))  
%         title('Regime: OFF')
%         grid on
%         figure(2)
%         scatter(asset, inventory, 2, control(:,2))  
%         title('Regime: ON')
%         grid on  
%        
%         
%     end    
    
end 
toc


% forward simulations 
rng(10)
simOutSample = 200000;

X0 = zeros(simOutSample ,1);
I0 =5;
demandMatrix = demandSimulate(alpha, K0, sigma, nstep, simOutSample, maturity, X0);

inventoryForward = zeros(simOutSample,nstep+1);
inventoryForward(:,1)=I0;

regimeForward = zeros(simOutSample,nstep+1);
regimeForward(:,1)=1;

Bt = zeros(simOutSample,nstep);
St = zeros(simOutSample,nstep);
dieselPower = zeros(simOutSample,nstep);
trueCost = zeros(simOutSample,nstep);
artificialCost = zeros(simOutSample,nstep);

costForward = zeros(simOutSample,nstep+1);

for iStep=1:1:nstep

    [~, ~, control, nextInventory, nextRegime, imbalance, batteryOutput] = oneStepOptimization_microgrid(demandMatrix(:,iStep) , inventoryForward(:,iStep),  Model{nstep-iStep+1});
    
    idx = sub2ind(size(nextRegime),(1:simOutSample).',regimeForward(:,iStep));

    inventoryForward(:,iStep+1) = nextInventory(idx);
    regimeForward(:,iStep+1) = nextRegime(idx);
    Bt(:,iStep) = batteryOutput(idx);
    St(:,iStep) = imbalance(idx);       
    dieselPower(:,iStep) = control(idx);      
    
    trueCost(:,iStep) = k(1)*(dieselPower(:,iStep).^0.9).*dt + switchCost.*(regimeForward(:,iStep+1)>regimeForward(:,iStep));
    artificialCost(:,iStep) = k(2).*St(:,iStep).*(St(:,iStep)>0).*dt - k(3).*St(:,iStep).*(St(:,iStep)<0).*dt;

end

penalty = -200*((inventoryForward(:,end)-I0)).*(inventoryForward(:,end)<I0);

costForward = trueCost + artificialCost;
pathWiseCost = sum(costForward,2) +penalty;
totalCost = mean(pathWiseCost);stdDeviation = std(pathWiseCost)/sqrt(simOutSample);
totalTrueCost = mean(sum(trueCost,2));
totalArtificialCost = mean(sum(artificialCost,2));



fprintf('nsim: %d x %d, regime=1,  totalCost= %d, totalTrueCost= %d, totalArtificialCost= %d \n', sites, batch, totalCost, totalTrueCost, totalArtificialCost);

end




