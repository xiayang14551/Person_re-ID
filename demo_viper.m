%% clean
close all;
clear;
clc;

%% setup
% uncomment the following line to download and prepare the dataset, when run demo_viper.m for the first time.
 setup_viper   % download and prepare the dataset

% To run for other datasets, insert the features in ..\data folder 
% and do the modifications in the code. 
params.dataset ='viper';   % the name of the dataset
workDir = pwd;
% loading the partitions that we used in our experiments
load(fullfile(workDir,'data',sprintf('%s_features.mat',params.dataset)));

%% set parameters
params.N = 316;  % the number of the persons included in training set
params.seed = 2;  % random partition seed

% initial params
params.Ktype = 'rbf';       % 'rbf' kernel function
params.dimReduc.dimType = 'kpca';       % dimensionality reduction method
params.dimReduc.ReducedDim = 315;
params.sigma = 1;   % sigma
params.lambda = 0.1;  
params.tao = 1;    % tao


%% CV-KCRC

XQDAdata = dataprepare(features,params);     % partition train/test set randomly for one trial

CV_KCRC(XQDAdata,params);    % run CV-KCRC for one trial

