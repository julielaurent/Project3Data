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

%% Gradual features and regression
i = 0;

 % PCA
[trainSet_norm, mu, sigma] = zscore(trainSet);
[coeff_pca, trainSet_pca, variance_pca] = pca(trainSet_norm);

testSet_pca = ((testSet - ones(size(testSet,1),1)*mu) ./ (ones(size(testSet,1),1)*sigma)) * coeff_pca;

for j = 50:50:950
    i = i + 1;
    
    trainSet_pca_it = trainSet_pca(:,1:j);
    testSet_pca_it = testSet_pca(:,1:j);
    
    % Regression
    trainI_X = ones(size(trainPosX,1),1);
    trainI_Y = ones(size(trainPosY,1),1);

    testI_X = ones(size(testPosX,1),1);
    testI_Y = ones(size(testPosY,1),1);


    trainFM = trainSet_pca_it;
    trainX_X_1 = [ trainI_X trainFM ];
    trainX_Y_1 = [ trainI_Y trainFM ];
    trainX_X_2 = [ trainI_X trainFM trainFM.^2 ];
    trainX_Y_2 = [ trainI_Y trainFM trainFM.^2 ];

    testFM = testSet_pca_it;
    testX_X_1 = [ testI_X testFM ];
    testX_Y_1 = [ testI_Y testFM ];
    testX_X_2 = [ testI_X testFM testFM.^2 ];
    testX_Y_2 = [ testI_Y testFM testFM.^2 ];

    b_X_1 = regress(trainPosX,trainX_X_1);
    b_Y_1 = regress(trainPosY,trainX_Y_1);
    b_X_2 = regress(trainPosX,trainX_X_2);
    b_Y_2 = regress(trainPosY,trainX_Y_2);

    % Error
    trainErrX_1(i) = immse(trainPosX,trainX_X_1*b_X_1);
    trainErrY_1(i) = immse(trainPosY,trainX_Y_1*b_Y_1);
    trainErrX_2(i) = immse(trainPosX,trainX_X_2*b_X_2);
    trainErrY_2(i) = immse(trainPosY,trainX_Y_2*b_Y_2);

    testErrX_1(i) = immse(testPosX,testX_X_1*b_X_1);
    testErrY_1(i) = immse(testPosY,testX_Y_1*b_Y_1);
    testErrX_2(i) = immse(testPosX,testX_X_2*b_X_2);
    testErrY_2(i) = immse(testPosY,testX_Y_2*b_Y_2);
end  

%% Figures
figure('Color','w');
% X
subplot(2,1,1);
title('Error on vector X');
hold on;
xlabel('Number of PCs');
ylabel('Error');
plot(50:50:950,trainErrX_1,'--b');
plot(50:50:950,trainErrX_2,'--r');
plot(50:50:950,testErrX_1,'-b');
plot(50:50:950,testErrX_2,'-r');
legend('Train error / Order 1','Train error / Order 2','Test error / Order 1','Test error / Order 2');
box off;
hold off;
% Y
subplot(2,1,2);
title('Error on vector Y');
hold on;
xlabel('Number of PCs');
ylabel('Error');
plot(50:50:950,trainErrY_1,'--b');
plot(50:50:950,trainErrY_2,'--r');
plot(50:50:950,testErrY_1,'-b');
plot(50:50:950,testErrY_2,'-r');
legend('Train error / Order 1','Train error / Order 2','Test error / Order 1','Test error / Order 2');
box off;
hold off;