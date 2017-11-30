clear all;
close all;
clc;

load('data.mat');

%% Partitioning

k = 0.05; % training set percentage of 5%

trainSet = Data(1:round(k*12862),:);
testSet = Data(round(k*12862)+1:end,:);
trainPosX = PosX(1:round(k*12862),:);
testPosX = PosX(round(k*12862)+1:end,:);
trainPosY = PosY(1:round(k*12862),:);
testPosY = PosY(round(k*12862)+1:end,:);

%% Linear Regression

trainI_X = ones(size(trainPosX,1),1);
trainI_Y = ones(size(trainPosY,1),1);

testI_X = ones(size(testPosX,1),1);
testI_Y = ones(size(testPosY,1),1);
 
trainFM = trainSet;
trainX_X_1 = [ trainI_X trainFM ];
trainX_Y_1 = [ trainI_Y trainFM ];
   
testFM = testSet;
testX_X_1 = [ testI_X testFM ];
testX_Y_1 = [ testI_Y testFM ];
 
% Regressor calculation
b_X_1 = regress(trainPosX,trainX_X_1);
b_Y_1 = regress(trainPosY,trainX_Y_1);

% Train and test errors        
trainErrX_1 = immse(trainPosX,trainX_X_1*b_X_1);
trainErrY_1 = immse(trainPosY,trainX_Y_1*b_Y_1);   
testErrX_1 = immse(testPosX,testX_X_1*b_X_1);
testErrY_1 = immse(testPosY,testX_Y_1*b_Y_1);

%% Plot

% X
figure('Color','w');
subplot(2,1,1);
title('Position Vector X');
hold on;
xlabel('Time');
ylabel('PosX');
plot(PosX,'-k','LineWidth',2);
plot(trainX_X_1*b_X_1,'--b');
plot(round(k*12862)+1:12862,testX_X_1*b_X_1,'-b');
legend('Real position vector','Regressed position vector (train set)','Regressed position vector (test set)');
box off;
axis([200 1200 -0.2 0.4]);
hold off;

% Y
subplot(2,1,2);
title('Position Vector Y');
hold on;
xlabel('Time');
ylabel('PosY');
plot(PosY,'-k','LineWidth',2);
plot(trainX_Y_1*b_Y_1,'--b');
plot(round(k*12862)+1:12862,testX_Y_1*b_Y_1,'-b');
box off;
axis([200 1200 -0.2 0.4]);
hold off;

