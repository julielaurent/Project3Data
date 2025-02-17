clear all;
close all;
clc;

load('data.mat');

%% Partitioning

k = 0.7; % training set percentage

trainSet = Data(1:round(k*12862),:);
testSet = Data(round(k*12862)+1:end,:);
trainPosX = PosX(1:round(k*12862),:);
testPosX = PosX(round(k*12862)+1:end,:);
trainPosY = PosY(1:round(k*12862),:);
testPosY = PosY(round(k*12862)+1:end,:);

[trainSet_norm, mu, sigma] = zscore(trainSet);
[coeff_pca, trainSet_pca, variance_pca] = pca(trainSet_norm);
figure()
imshow(cov(trainSet));
figure()
imshow(cov(trainSet_pca));

%testSet_pca = ((testSet' - mu') ./ sigma')' * coeff_pca;
testSet_pca = ((testSet - ones(size(testSet,1),1)*mu) ./ (ones(size(testSet,1),1)*sigma)) * coeff_pca;

%% Cumulative variance
% First element representes the percentage of variance explained by PC1
% Second element represents the percentage of variance explained by the 2
% first PCs (PC1 & PC2)
% And so on...
VarCumPercentage = cumsum(variance_pca)/sum(variance_pca)*100;
figure('Color','w');
title('Number of PCA Explaining 90% of the Total Variance of the Data');
plot(VarCumPercentage);
xlabel('Principal Components');
ylabel('Cumulative Variance Explained [%]');
box off;
axis([0 960 0 100]);
line([0 960], [90 90],'Color','r');
line([741 741],[0 90],'Color','r','LineStyle','--');

%741 features explain 90.0268 % of the variances

%% Regression

trainI_X = ones(size(trainPosX,1),1);
trainI_Y = ones(size(trainPosY,1),1);

testI_X = ones(size(testPosX,1),1);
testI_Y = ones(size(testPosY,1),1);

    
trainFM = trainSet_pca;
trainX_X_1 = [ trainI_X trainFM ];
trainX_Y_1 = [ trainI_Y trainFM ];
trainX_X_2 = [ trainI_X trainFM trainFM.^2 ];
trainX_Y_2 = [ trainI_Y trainFM trainFM.^2 ];
   
testFM = testSet_pca;
testX_X_1 = [ testI_X testFM ];
testX_Y_1 = [ testI_Y testFM ];
testX_X_2 = [ testI_X testFM testFM.^2 ];
testX_Y_2 = [ testI_Y testFM testFM.^2 ];
       
b_X_1 = regress(trainPosX,trainX_X_1);
b_Y_1 = regress(trainPosY,trainX_Y_1);
b_X_2 = regress(trainPosX,trainX_X_2);
b_Y_2 = regress(trainPosY,trainX_Y_2);
        
trainErrX_1 = immse(trainPosX,trainX_X_1*b_X_1);
trainErrY_1 = immse(trainPosY,trainX_Y_1*b_Y_1);
trainErrX_2 = immse(trainPosX,trainX_X_2*b_X_2);
trainErrY_2 = immse(trainPosY,trainX_Y_2*b_Y_2);
    
testErrX_1 = immse(testPosX,testX_X_1*b_X_1);
testErrY_1 = immse(testPosY,testX_Y_1*b_Y_1);
testErrX_2 = immse(testPosX,testX_X_2*b_X_2);
testErrY_2 = immse(testPosY,testX_Y_2*b_Y_2);

%% Figures

% Order 1
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
axis([8800 9200 -0.05 0.2]);
hold off;

% Order 1
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
axis([8800 9200 0.15 0.3]);
hold off;

% Order 2
% X
figure('Color','w');
title('Position Vector X');
hold on;
xlabel('Time');
ylabel('PosX');
plot(PosX,'-k','LineWidth',2);
plot(trainX_X_2*b_X_2,'--r');
plot(round(k*12862)+1:12862,testX_X_2*b_X_2,'-r');
legend('Real X','Regressed X (train set)','Regressed X (test set)');
box off;
axis([8800 9200 -0.05 0.2]);
hold off;

% Order 2
% Y
figure('Color','w');
title('Position Vector Y');
hold on;
xlabel('Time');
ylabel('PosY');
plot(PosY,'-k','LineWidth',2);
plot(trainX_Y_2*b_Y_2,'--r');
plot(round(k*12862)+1:12862,testX_Y_2*b_Y_2,'-r');
legend('Real Y','Regressed Y (train set)','Regressed Y (test set)');
box off;
axis([8800 9200 0.15 0.3]);
hold off;

% Comparison order 1 and 2
% X
figure('Color','w');
subplot(2,1,1);
title('Position Vector X');
hold on;
xlabel('Time');
ylabel('PosX');
plot(PosX,'-k','LineWidth',2);
plot(round(k*12862)+1:12862,testX_X_1*b_X_1,'-b');
plot(round(k*12862)+1:12862,testX_X_2*b_X_2,'-r');
legend('Real position vector','Linear regression on position vector (test set)','Second order regression on position vector (test set)');
box off;
axis([10100 10500 -0.05 0.2]);
hold off;

% Comparison order 1 and 2
% Y
subplot(2,1,2);
title('Position Vector Y');
hold on;
xlabel('Time');
ylabel('PosY');
plot(PosY,'-k','LineWidth',2);
plot(round(k*12862)+1:12862,testX_Y_1*b_Y_1,'-b');
plot(round(k*12862)+1:12862,testX_Y_2*b_Y_2,'-r');
box off;
axis([10100 10500 0.15 0.3]);
hold off;