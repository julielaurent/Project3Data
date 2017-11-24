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

testSet_pca = ((testSet' - mu') ./ sigma')' * coeff_pca;

%covMat = cov(trainSet_pca);
%imshow(covMat);

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