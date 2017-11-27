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

% L1 normalization
%trainSet_norm = norm(trainSet,1);

%% Lasso
lambda = logspace(-10,0,15);

b_X = lasso(trainSet, trainPosX);
b_Y = lasso(trainSet, trainPosY);
