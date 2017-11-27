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

[trainSet_norm, mu, sigma] = zscore(trainSet);
[coeff_pca, trainSet_pca, variance_pca] = pca(trainSet_norm);

testSet_pca = ((testSet' - mu') ./ sigma')' * coeff_pca;