function [totalCost,totalTrueCost,totalArtificialCost] = microgrid_Adaptivegp2d()
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
global degree; degree = 3;

maturity  = 24*2; % in hours
global dt; dt = 15/60;
nstep = fix(maturity/dt); % 

% stock
alpha = 0.5; K0 = 0; sigma = 2; processType = 'Regular';


%% backward simulation

rng('shuffle')


load('demandDataForProbabilityPenaltyNew.mat');
load('InventoryDataForProbabilityPenaltyNew.mat');
load('RegimeDataForProbabilityPenaltyNew.mat');

lhs = lhsdesign(0.4*sites,2);
% X0 = -10 + 20*lhs(:,1);
X0 = -7 + 14*lhs(:,1);
ICord = 10*lhs(:,2);

offIndx=XM(:,end)==1;
W1 = [datasample([XP(offIndx,end-1) XI(offIndx,end)],0.6*sites);[X0,ICord]];

onIndx=XM(:,end)==2;
W2 = [datasample([XP(onIndx,end-1) XI(onIndx,end)],0.6*sites);[X0,ICord]];

X01 = W1(:,1); ICord1 = W1(:,2);
X02 = W2(:,1); ICord2 = W2(:,2);


ICord1_rep = repelem(ICord1,batch);
X01_rep = repelem(X01,batch);

ICord2_rep = repelem(ICord2,batch);
X02_rep = repelem(X02,batch);

demandMatrix1 = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X01_rep);
demandMatrix2 = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X02_rep);

finalCost1 = -200*((ICord1_rep-5)).*(ICord1_rep<5); %2*0*demandMatrix(:,end).*(ICord_rep-5).*(ICord_rep<5);
finalCost1 = mean(reshape(finalCost1,[batch,sites]),1)';

finalCost2 = -200*((ICord2_rep-5)).*(ICord2_rep<5); %2*0*demandMatrix(:,end).*(ICord_rep-5).*(ICord_rep<5);
finalCost2 = mean(reshape(finalCost2,[batch,sites]),1)';

costNext = zeros(sites,no_regime);
costNext(:,1) = finalCost1; costNext(:,2) = finalCost2;


modelIndx = 1;

tic
for iStep = nstep:-1:1

    gprMdl{1} = compact(fitrgp([X01, ICord1],costNext(:,1), 'KernelFunction','ardsquaredexponential'));
    gprMdl{2} = compact(fitrgp([X02, ICord2],costNext(:,2), 'KernelFunction','ardsquaredexponential'));

  
    
    if iStep<100
        dIndx = 99;
    else
        dIndx = iStep;
    end
      
    lhs = lhsdesign(0.4*sites,2);
%     X0 = -10 + 20*lhs(:,1);
    X0 = -7 + 14*lhs(:,1);
    ICord = 10*lhs(:,2);

    offIndx=XM(:,dIndx)==1;
    W1 = [datasample([XP(offIndx,dIndx-1) XI(offIndx,dIndx)],0.6*sites);[X0,ICord]];

    onIndx=XM(:,dIndx)==2;
    W2 = [datasample([XP(onIndx,dIndx-1) XI(onIndx,dIndx)],0.6*sites);[X0,ICord]];

    X01 = W1(:,1); ICord1 = W1(:,2);
    X02 = W2(:,1); ICord2 = W2(:,2);

    ICord1_rep = repelem(ICord1,batch);
    X01_rep = repelem(X01,batch);

    ICord2_rep = repelem(ICord2,batch);
    X02_rep = repelem(X02,batch);

    demandMatrix1 = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X01_rep);
    demandMatrix2 = demandSimulate(alpha, K0, sigma, 1, nsim, dt, X02_rep); 
            
    [cost1, ~, ~, ~, ~, ~, ~] = oneStepOptimization_microgrid(demandMatrix1(:,2), ICord1_rep, gprMdl);
    [cost2, ~, ~, ~, ~, ~, ~] = oneStepOptimization_microgrid(demandMatrix2(:,2), ICord2_rep, gprMdl);

    
    costNext(:,1) = mean(reshape(cost1(:,1),[batch,sites]),1)';
    costNext(:,2) = mean(reshape(cost2(:,2),[batch,sites]),1)';
    
    Model{modelIndx} = gprMdl;
    modelIndx=modelIndx+1;    


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




