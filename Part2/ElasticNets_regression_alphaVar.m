clear all;
close all;
clc;

load('Data.mat');

%% Partitioning and normalization

k = 0.05; % training set percentage of 5%

trainSet = Data(1:round(k*12862),:);
testSet = Data(round(k*12862)+1:end,:);
trainPosX = PosX(1:round(k*12862),:);
testPosX = PosX(round(k*12862)+1:end,:);
trainPosY = PosY(1:round(k*12862),:);
testPosY = PosY(round(k*12862)+1:end,:);


%% Elastic Net
lambda = logspace(-10,0,15);
i = 0;
for alpha = 0.1:0.1:1
    i = i + 1;
    [B_X, FitInfo_X] = lasso(trainSet, trainPosX, 'CV', 10, 'Lambda', lambda, 'Alpha', alpha);
    [B_Y, FitInfo_Y] = lasso(trainSet, trainPosY, 'CV', 10, 'Lambda', lambda, 'Alpha', alpha);

    % Number of non-zero weights
%     Nb_nonzero_X = FitInfo_X.DF;
%     Nb_nonzero_Y = FitInfo_Y.DF;

%     % Plot CV MSE for each lambda
%     figure('Color','w');
%     semilogx(lambda,FitInfo_X.MSE,lambda,FitInfo_Y.MSE);
%     xlabel('Lambda');
%     ylabel('MSE');
%     legend('Position vector X','Position vector Y')
%     title('CV MSE for each Lambda');
%     box off;

    % Lambda with best MSE
    best_nb_lambda_X(i) = FitInfo_X.IndexMinMSE;
    best_nb_lambda_Y(i) = FitInfo_Y.IndexMinMSE;
    best_lambda_X(i) = FitInfo_X.LambdaMinMSE;
    best_lambda_Y(i) = FitInfo_Y.LambdaMinMSE;
    
    MSE_minLambda_X(i) = min(FitInfo_X.MSE);
    MSE_minLambda_Y(i) = min(FitInfo_Y.MSE);

end

% Best lambda and alpha
alpha = 0.1:0.1:1;
idx_bestAlpha_X = find(MSE_minLambda_X == min(MSE_minLambda_X));
idx_bestAlpha_Y = find(MSE_minLambda_Y == min(MSE_minLambda_Y));
min_MSE_X = min(MSE_minLambda_X);
min_MSE_Y = min(MSE_minLambda_Y);
best_alpha_X = alpha(idx_bestAlpha_X);
bestAssociated_lambda_X = best_lambda_X(idx_bestAlpha_X);
best_alpha_Y = alpha(idx_bestAlpha_Y);
bestAssociated_lambda_Y = best_lambda_Y(idx_bestAlpha_Y);

% Elastic net with optimized parameters
[B_X_opt, FitInfo_X_opt] = lasso(trainSet, trainPosX, 'CV', 10, 'Lambda', bestAssociated_lambda_X, 'Alpha', best_alpha_X);
[B_Y_opt, FitInfo_Y_opt] = lasso(trainSet, trainPosY, 'CV', 10, 'Lambda', bestAssociated_lambda_Y, 'Alpha', best_alpha_Y);

Nb_nonzero_X = FitInfo_X_opt.DF;
Nb_nonzero_Y = FitInfo_Y_opt.DF;

% Regression
Test_regressed_X = testSet * B_X + FitInfo_X.Intercept;
Test_regressed_Y = testSet * B_Y + FitInfo_Y.Intercept;

% Plot regressed data
%X
figure('Color','w');
subplot(2,1,1);
title('Position Vector X');
hold on;
xlabel('Time');
ylabel('PosX');
plot(PosX,'-k','LineWidth',2);
plot(round(k*12862)+1:12862,Test_regressed_X,'-b');
legend('Real position vector','Regressed position vector (test set)');
box off;
axis([8900 9100 -0.05 0.2]);
hold off;

% Y
subplot(2,1,2);
title('Position Vector Y');
hold on;
xlabel('Time');
ylabel('PosY');
plot(PosY,'-k','LineWidth',2);
plot(round(k*12862)+1:12862,Test_regressed_Y,'-b');
box off;
axis([8900 9100 0.15 0.3]);
hold off;

% Test MSE
testErrX = immse(testPosX,Test_regressed_X);
testErrY = immse(testPosY,Test_regressed_Y);