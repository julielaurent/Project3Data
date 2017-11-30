clear all;
close all;
clc;

load('data.mat');

%% Partitioning and normalization

k = 0.05; % training set percentage of 5%

trainSet = Data(1:round(k*12862),:);
testSet = Data(round(k*12862)+1:end,:);
trainPosX = PosX(1:round(k*12862),:);
testPosX = PosX(round(k*12862)+1:end,:);
trainPosY = PosY(1:round(k*12862),:);
testPosY = PosY(round(k*12862)+1:end,:);

%% Lasso
lambda = logspace(-10,0,15);
alpha = 0.5;

[B_X_L, FitInfo_X_L] = lasso(trainSet, trainPosX, 'CV', 10, 'Lambda', lambda);
[B_Y_L, FitInfo_Y_L] = lasso(trainSet, trainPosY, 'CV', 10, 'Lambda', lambda);
[B_X_E, FitInfo_X_E] = lasso(trainSet, trainPosX, 'CV', 10, 'Lambda', lambda, 'Alpha', alpha);
[B_Y_E, FitInfo_Y_E] = lasso(trainSet, trainPosY, 'CV', 10, 'Lambda', lambda, 'Alpha', alpha);

% Lambda with best MSE
best_nb_lambda_X_L = FitInfo_X_L.IndexMinMSE;
best_nb_lambda_Y_L = FitInfo_Y_L.IndexMinMSE;

best_nb_lambda_X_E = FitInfo_X_E.IndexMinMSE;
best_nb_lambda_Y_E = FitInfo_Y_E.IndexMinMSE;

% Regression
Test_regressed_X_L = testSet * B_X_L(:,best_nb_lambda_X_L) + FitInfo_X_L.Intercept(best_nb_lambda_X_L);
Test_regressed_Y_L = testSet * B_Y_L(:,best_nb_lambda_Y_L) + FitInfo_Y_L.Intercept(best_nb_lambda_Y_L);

Test_regressed_X_E = testSet * B_X_E(:,best_nb_lambda_X_E) + FitInfo_X_E.Intercept(best_nb_lambda_X_E);
Test_regressed_Y_E = testSet * B_Y_E(:,best_nb_lambda_Y_E) + FitInfo_Y_E.Intercept(best_nb_lambda_Y_E);

% Plot regressed data
%X
figure('Color','w');
subplot(2,1,1);
title('Position Vector X');
hold on;
xlabel('Time');
ylabel('PosX');
plot(PosX,'-k','LineWidth',2);
plot(round(k*12862)+1:12862,Test_regressed_X_L,'-b');
plot(round(k*12862)+1:12862,Test_regressed_X_E,'-r');
legend('Real position vector','Regressed position vector with Lasso Regularization (test set)','Regressed position vector with Elastic nets Regularization (test set)');
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
plot(round(k*12862)+1:12862,Test_regressed_Y_L,'-b');
plot(round(k*12862)+1:12862,Test_regressed_Y_E,'-r');
box off;
axis([8900 9100 0.15 0.3]);
hold off;

