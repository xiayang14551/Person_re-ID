function CV_KCRC(XQDAdata,params)

% set parameters for dimensionality reduction process  
params.dimReduc.Kernel = 1;       % kernelize or not
params.dimReduc.Regu = 1;      % regularize or not
params.dimReduc.ReguAlpha = 0.001;    % (A+ReguAlpha*I) is used to replace singular matrix A
params.dimReduc.constructW.NeighborMode = 'Supervised'; % only used in KLPP
params.dimReduc.constructW.WeightMode = 'Binary';  %  'Binary' or 'HeatKernel' or 'Cosine' only used in KLPP
% params.dimReduc.constructW.bLDA = 1;
params.dimReduc.constructW.bNormalized = 1;
params.dimReduc.constructW.gnd = [];
% params.dimReduc.constructW.t = 0.8;


    xQDA_feat = XQDAdata{1};
    test_xQDA_feat = XQDAdata{2};
    params.idxtrain = XQDAdata{3};
    params.idxtest = XQDAdata{4};
    
    train_a_ker = constructKernel(xQDA_feat.dataA ,xQDA_feat.dataA,params);
    test_a_ker =  constructKernel(xQDA_feat.dataA ,test_xQDA_feat.dataA,params);
    train_b_ker = constructKernel(xQDA_feat.dataB ,xQDA_feat.dataB,params);
    test_b_ker =  constructKernel(xQDA_feat.dataB ,test_xQDA_feat.dataB,params);
    
    params.dimReduc.constructW.gnd = [];
    if params.dimReduc.dimType == 'klpp'
        params.dimReduc.constructW.gnd = (params.idxtrain)';
        params.Wy = constructW(xQDA_feat.dataA,params.dimReduc.constructW);
        params.Wx = constructW(xQDA_feat.dataB,params.dimReduc.constructW);
    else
        params.Wy = [];
        params.Wx = [];
    end
    % ----------------- CV-KCRC Method --------------------
    
    % CV-KCRC
    tic
    Model = CV_KCRC_fun(train_a_ker,train_b_ker, params);
    
    dX = Model.AlfaX.BetaX*test_a_ker;
    dY = Model.AlfaY.BetaX*test_a_ker;
    Alfax =  Model.AlfaX.BetaY*test_b_ker;
    Alfay =  Model.AlfaY.BetaY*test_b_ker;
    
    parfor i=1:numel(params.idxtest) % For each probe
        fprintf('Processing image %d \n',i);
        alfa_error = zeros(1,numel(params.idxtest));
        for n=1:numel(params.idxtest) % For each gallery
            alfax =  dX(:,i)+Alfax(:,n);
            alfay =  dY(:,i)+Alfay(:,n);
            % computing the cosine distance between coding vectors
            alfa_error(n) = pdist2(alfax',alfay','cosine');
        end
        [~,idx_error(i,:)] = sort(alfa_error,'ascend');
    end
    
    resp = zeros(1,size(test_b_ker,2));
    for i=1:numel(params.idxtest) % For each probe
        resp(idx_error(i,:)==i) = resp(idx_error(i,:)==i) + 1;
    end
    rank_k = cumsum(resp)./size(test_a_ker,2);
    pos = [1 5 10 20 30];
    fprintf('Time is: %d  s\n',toc);
    fprintf('Ranking results of CV-KCRC: %f , %f , %f, %f , %f \n', rank_k(pos));
    
% plot 
plot(1:30,rank_k(1:30));
%axis labels
title('CMC Curve');
xlabel('Rank')
ylabel('Recognition Rate')
end


function Model = CV_KCRC_fun(KX,KY,params)
% Cross-View Kernel Collaborative Representation Classification (refer to the paper for more details).

params.W = params.Wx;
[Ax,~] = DimReduce(KX,params);
params.W = params.Wy;
[Ay,~] = DimReduce(KY,params);

temp0 = KX'*(Ax*Ax')*KX;
temp1 = KY'*(Ay*Ay')*KY;
n = size(temp0,1);
m = size(temp1,1);
% 
Sx = inv(temp0 + (params.lambda + params.tao)*eye(n));
Sy = inv(temp1 + (params.lambda + params.tao)*eye(m));

% 
Ux = inv(eye(n) -  params.tao^2*Sx*Sy);
Uy = inv(eye(m) -  params.tao^2*Sy*Sx);
Qx = KX'*(Ax*Ax');
Qy = KY'*(Ay*Ay');
% Computing the projection matrices
% BetaX
BetaX = Ux*Sx*Qx;
BetaY = params.tao*Ux*Sx*Sy*Qy;
Model.AlfaX.BetaX = BetaX;
Model.AlfaX.BetaY = BetaY;
% BetaY
BetaY1 = Uy*Sy*Qy;
BetaX1 = params.tao*Uy*Sy*Sx*Qx;
Model.AlfaY.BetaX = BetaX1;
Model.AlfaY.BetaY = BetaY1;
end

% Kernel Function
function k = constructKernel(x,y,param)
% construct Kernel Matrix.
x = x';y = y';
switch param.Ktype
    case 'rbf'
        k = kernelRBF(x,y,param);
    case 'linear'
        k = kernelLiner(x,y);
    case 'poly2'
        param.poly.Degree = 2;
        k = kernelPoly(x,y,param);
    case 'poly3'
        param.poly.Degree = 3;
        k = kernelPoly(x,y,param);
    otherwise
        fprint('Kernel type unsupport !');
end
end
function k = kernelRBF(x,y,params)
% rbf kernel rbf(x,y)=exp((-(1/sigma^2).*(|x-y|.^2));
% Usage:
% k=kernelRBF(x,y)
% k=kernelRBF(x,y,param)
% x,y: column vectors, or matrices.
% param: sigma, [sigma].
% k, sigma or matrix, the kernel values
% Yifeng Li, May 26, 2011.
if nargin<3
    sigma=1;
else
    sigma=params.sigma;
    if sigma==0
        error('sigma must not be zero!');
    end
end
k=exp((-(1/sigma^2)).*(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1)));
end
function k = kernelPoly(x,y,params)    %   k = [param(1)*(x'*y)+param(2)]^param(3)
% Polynomial kernel k=(Gamma.*(x'*y)+ Coefficient).^Degree;
% Usage:
% k=kernelPoly(x,y)
% k=kernelPoly(x,y,param)
% x,y: column vectors, or matrices.
% param: [Gamma;Coefficient;Degree]
% k, scalar or matrix, the kernel values
% Yifeng Li, May 26, 2011.
if nargin<3
    Gamma=1;
    Coefficient=0;
    Degree=2;
else
    Gamma=params.poly.Gamma;
    Coefficient=params.poly.Coefficient;
    Degree=params.poly.Degree;
end
k=(Gamma.*(x'*y)+ Coefficient).^Degree;
end
function k = kernelLiner(x,y)
% Liner kernel   kernelLiner(x,y)=x'*y;
% Usage:
% k=kernelLiner(x,y)
% x,y: column vectors, or matrices.
% k, scalar or matrix, the kernel values
k=x'*y;
end
% Dimensionality Reduction
function [eigvec,eigvalu] = DimReduce(data,para)
label = (para.idxtrain)';
switch para.dimReduc.dimType
    case 'kpca'
        [eigvec,eigvalu] = KPCA(data,para.dimReduc);
    case 'klda'
        [eigvec,eigvalu] = KDA(para.dimReduc,label,data);
    case 'klpp'
        [eigvec,eigvalu] = KLPP(para.W,para.dimReduc,data);
    otherwise
        fprint('Dim reduction type unsupport !');
end
end
function [eigvector, eigvalue] = KPCA(data, options)
%KPCA	Kernel Principal Component Analysis
%	Usage:
%       [eigvector, eigvalue] = KPCA(data, options)
%             Input:
%               data    -
%                      if options.Kernel = 0
%                           Data matrix. Each row vector of fea is a data
%                           point.
%                      if options.Kernel = 1
%                           Kernel matrix.
%             options   - Struct value in Matlab. The fields in options that can be set:
%                      Kernel  -  1: data is actually the kernel matrix.
%                                      0: ordinary data matrix.
%                                          Default: 0
%        Please see constructKernel.m for other Kernel options.
%                     ReducedDim - The dimensionality of the reduced subspace. If 0,
%                                             all the dimensions will be kept. Default is 30.
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%	Examples:
%           options.KernelType = 'Gaussian';
%           options.t = 1;
%           options.ReducedDim = 4;
% 			fea = rand(7,10);
% 			[eigvector,eigvalue] = KPCA(fea,options);
%           feaTest = rand(3,10);
%           Ktest = constructKernel(feaTest,fea,options)
%           Y = Ktest*eigvector;
%Reference:
%   Bernhard Schölkopf, Alexander Smola, Klaus-Robert Müller, “Nonlinear
%   Component Analysis as a Kernel Eigenvalue Problem", Neural Computation,
%   10:1299-1319, 1998.
%   version 1.1 --Dec./2011
%   version 1.0 --April/2005
%   Written by Deng Cai (dengcai AT gmail.com)

MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power

ReducedDim = 30;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end

if isfield(options,'Kernel') && options.Kernel
    K = data;
else
    K = constructKernel(data,[],options);
end
clear data;

nSmp = size(K,1);
if (ReducedDim > nSmp) || (ReducedDim <=0)
    ReducedDim = nSmp;
end

sumK = sum(K,2);
H = repmat(sumK./nSmp,1,nSmp);
K = K - H - H' + sum(sumK)/(nSmp^2);
K = max(K,K');
clear H;

if nSmp > MAX_MATRIX_SIZE && ReducedDim < nSmp*EIGVECTOR_RATIO
    % using eigs to speed up!
    option = struct('disp',0);
    [eigvector, eigvalue] = eigs(K,ReducedDim,'la',option);
    eigvalue = diag(eigvalue);
else
    [eigvector, eigvalue] = eig(K);
    eigvalue = diag(eigvalue);
    
    [dump, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:,index);
end

if ReducedDim < length(eigvalue)
    eigvalue = eigvalue(1:ReducedDim);
    eigvector = eigvector(:, 1:ReducedDim);
end

maxEigValue = max(abs(eigvalue));
eigIdx = find(abs(eigvalue)/maxEigValue < 1e-6);
eigvalue (eigIdx) = [];
eigvector (:,eigIdx) = [];

for i=1:length(eigvalue) % normalizing eigenvector
    eigvector(:,i)=eigvector(:,i)/sqrt(eigvalue(i));
end

end
function [eigvector, eigvalue] = KDA(options,label,data)
% KDA: Kernel Discriminant Analysis
%       [eigvector, eigvalue] = KDA(options, label, data)
%             Input:
%               data    -
%                      if options.Kernel = 0
%                           Data matrix. Each row vector of fea is a data point.
%                      if options.Kernel = 1
%                           Data matrix is Kernel matrix.
%               label   - Colunm vector of the label information for each
%                       data point.
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                           Kernel  -  1: data is actually the kernel matrix.
%                                      0: ordinary data matrix.
%                                      Default: 0
%
%                            Regu - 1: regularized solution,
%                                        a* = argmax (a'KWKa)/(a'KKa+ReguAlpha*I)
%                                      0: solve the sinularity problem by SVD
%                                      Default: 0
%                         ReguAlpha -  The regularization parameter. Valid
%                                      when Regu==1. Default value is 0.001.
%                       Please see constructKernel.m for other Kernel options.
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%               eigvalue  - The sorted eigvalue of LDA eigen-problem.
%               elapse    - Time spent on different steps
%    Examples:
%       fea = rand(50,70);
%       label = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options.KernelType = 'Gaussian';
%       options.t = 1;
%       [eigvector, eigvalue] = KDA(options, label, fea);
%       feaTest = rand(3,10);
%       Ktest = constructKernel(feaTest,fea,options)
%       Y = Ktest*eigvector;
% See also KSR, KLPP, KGE
% NOTE:
% In paper [2], we present an efficient approach to solve the optimization
% problem in KDA. We named this approach as Kernel Spectral Regression
% (KSR). I strongly recommend using KSR instead of this KDA algorithm.
%Reference:
%   [1] G. Baudat, F. Anouar, “Generalized
%   Discriminant Analysis Using a Kernel Approach", Neural Computation,
%   12:2385-2404, 2000.
%   [2] Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%   [3] Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.
%   version 3.0 --Dec/2011
%   version 2.0 --August/2007
%   version 1.0 --April/2005
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
if (~exist('options','var'))
    options = [];
end

if ~isfield(options,'Regu') || ~options.Regu
    bPCA = 1;
else
    bPCA = 0;
    if ~isfield(options,'ReguAlpha')
        options.ReguAlpha = 0.001;
    end
end

if isfield(options,'Kernel') && options.Kernel
    K = data;
    K = max(K,K');
else
    K = constructKernel(data,[],options);
end
clear data;

% ====== Initialization
nSmp = size(K,1);
if length(label) ~= nSmp
    error('label and data mismatch!');
end

classLabel = unique(label);
nClass = length(classLabel);
Dim = nClass - 1;

K_orig = K;

sumK = sum(K,2);
H = repmat(sumK./nSmp,1,nSmp);
K = K - H - H' + sum(sumK)/(nSmp^2);
K = max(K,K');
clear H;

%======================================
% SVD
%======================================
if bPCA
    
    [U,D] = eig(K);
    D = diag(D);
    
    maxEigValue = max(abs(D));
    eigIdx = find(abs(D)/maxEigValue < 1e-6);
    if length(eigIdx) < 1
        [dump,eigIdx] = min(D);
    end
    D (eigIdx) = [];
    U (:,eigIdx) = [];
    
    Hb = zeros(nClass,size(U,2));
    for i = 1:nClass
        index = find(label==classLabel(i));
        classMean = mean(U(index,:),1);
        Hb (i,:) = sqrt(length(index))*classMean;
    end
    
    [dumpVec,eigvalue,eigvector] = svd(Hb,'econ');
    eigvalue = diag(eigvalue);
    
    if length(eigvalue) > Dim
        eigvalue = eigvalue(1:Dim);
        eigvector = eigvector(:,1:Dim);
    end
    eigvector =  (U.*repmat((D.^-1)',nSmp,1))*eigvector;
else
    Hb = zeros(nClass,nSmp);
    for i = 1:nClass
        index = find(label==classLabel(i));
        classMean = mean(K(index,:),1);
        Hb (i,:) = sqrt(length(index))*classMean;
    end
    B = Hb'*Hb;
    T = K*K;
    
    for i=1:size(T,1)
        T(i,i) = T(i,i) + options.ReguAlpha;
    end
    
    B = double(B);
    T = double(T);
    B = max(B,B');
    T = max(T,T');
    
    option = struct('disp',0);
    [eigvector, eigvalue] = eigs(B,T,Dim,'la',option);
    eigvalue = diag(eigvalue);
end

tmpNorm = sqrt(sum((eigvector'*K_orig).*eigvector',2));
eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1);
eigvector = eigvector(:,1:options.ReducedDim);
end
function [eigvector, eigvalue] = KLPP(W, options, data)
% KLPP: Kernel Locality Preserving Projections
%       [eigvector, eigvalue] = KLPP(W, options, data)
%             Input:
%               data    -
%                      if options.Kernel = 0
%                           Data matrix. Each row vector of fea is a data
%                           point.
%                      if options.Kernel = 1
%                           Kernel matrix.
%               W       - Affinity matrix. You can either call "constructW"
%                         to construct the W, or construct it by yourself.
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                         Please see KGE.m for other options.
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%               eigvalue  - The sorted eigvalue of LPP eigen-problem.
%               elapse    - Time spent on different steps
%    Examples:
%       fea = rand(50,10);
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%       options.Regu = 0;
%       [eigvector, eigvalue] = KLPP(W, options, fea);
%       feaTest = rand(5,10);
%       Ktest = constructKernel(feaTest,fea,options)
%       Y = Ktest*eigvector;
%-----------------------------------------------------------------
%       fea = rand(50,10);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.bLDA = 1;
%       W = constructW(fea,options);
%
%       options.KernelType = 'Gaussian';
%       options.t = 1;
%       options.Regu = 1;
%       options.ReguAlpha = 0.01;
%       [eigvector, eigvalue] = KLPP(W, options, fea);
%       feaTest = rand(5,10);
%       Ktest = constructKernel(feaTest,fea,options)
%       Y = Ktest*eigvector;
%-----------------------------------------------------------------%
% See also constructW, KGE, KDA, KSR
% NOTE:
% In paper [3], we present an efficient approach to solve the optimization
% problem in KLPP. We named this approach as Kernel Spectral Regression
% (KSR). I strongly recommend using KSR instead of this KLPP algorithm.
%
%Reference:
%	[1] Xiaofei He, and Partha Niyogi, "Locality Preserving Projections"
%	Advances in Neural Information Processing Systems 16 (NIPS 2003),
%	Vancouver, Canada, 2003.
%	[2] Xiaofei He, "Locality Preserving Projections"
%	PhD's thesis, Computer Science Department, The University of Chicago,
%	2005.
%   [3] Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for
%   Dimensionality Reduction", Department of Computer Science
%   Technical Report No. 2856, University of Illinois at Urbana-Champaign
%   (UIUCDCS-R-2007-2856), May 2007.
%   version 2.0 --May/2007
%   version 1.0 --April/2004
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)

if (~exist('options','var'))
    options = [];
end

if isfield(options,'Kernel') && options.Kernel
    K = data;
    clear data;
else
    K = constructKernel(data,[],options);
end

nSmp = size(K,1);
D = full(sum(W,2));
if isfield(options,'Regu') && options.Regu
    options.ReguAlpha = options.ReguAlpha*sum(D)/length(D);
end
D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);

options.Kernel = 1;
[eigvector, eigvalue] = KGE(W, D, options, K);

eigIdx = find(eigvalue < 1e-3);
eigvalue (eigIdx) = [];
eigvector(:,eigIdx) = [];
end
function [eigvector, eigvalue] = KGE(W, D, options, data)
% KGE: Kernel Graph Embedding
%       [eigvector, eigvalue] = KGE(W, D, options, data)
%             Input:
%               data    -
%                      if options.Kernel = 0
%                           Data matrix. Each row vector of fea is a data
%                           point.
%                      if options.Kernel = 1
%                           Kernel matrix.
%               W       - Affinity graph matrix.
%               D       - Constraint graph matrix.
%                         KGE solves the optimization problem of
%                         a* = argmax (a'KWKa)/(a'KDKa)
%                         Default: D = I
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%                          Kernel  -  1: data is actually the kernel matrix.
%                                     0: ordinary data matrix.  Default: 0
%                     ReducedDim   -  The dimensionality of the reduced subspace. If 0, all the dimensions
%                                                will be kept. Default is 30.
%                            Regu  -  1: regularized solution,
%                                          a* = argmax (a'KWKa)/(a'KDKa+ReguAlpha*I)
%                                          0: solve the sinularity problem by SVD. Default: 0
%                        ReguAlpha -  The regularization parameter. Valid
%                                              when Regu==1. Default value is 0.01.
%  Please see constructKernel.m for other Kernel options.
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = K(x,:)*eigvector
%                           will be the embedding result of x.
%                           K(x,:) = [K(x1,x),K(x2,x),...K(xm,x)]
%               eigvalue  - The sorted eigvalue of the eigen-problem.
%               elapse    - Time spent on different steps
%  Examples:  See also KernelLPP, KDA, constructKernel.
%  Reference:
%   1. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for Efficient
%   Regularized Subspace Learning", IEEE International Conference on
%   Computer Vision (ICCV), Rio de Janeiro, Brazil, Oct. 2007.
%
%   2. Deng Cai, Xiaofei He, Yuxiao Hu, Jiawei Han, and Thomas Huang,
%   "Learning a Spatially Smooth Subspace for Face Recognition", CVPR'2007
%
%   3 Deng Cai, Xiaofei He, and Jiawei Han. "Speed Up Kernel Discriminant
%   Analysis", The VLDB Journal, vol. 20, no. 1, pp. 21-33, January, 2011.
%
%   4. Deng Cai, "Spectral Regression: A Regression Framework for
%   Efficient Regularized Subspace Learning", PhD Thesis, Department of
%   Computer Science, UIUC, 2009.
%   version 3.0 --Dec/2011
%   version 2.0 --July/2007
%   version 1.0 --Sep/2006
%   Written by Deng Cai (dengcai AT gmail.com)
MAX_MATRIX_SIZE = 1600; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power
if (~exist('options','var'))
    options = [];
end

if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
else
    ReducedDim = 30;
end

if ~isfield(options,'Regu') || ~options.Regu
    bPCA = 1;
else
    bPCA = 0;
    if ~isfield(options,'ReguAlpha')
        options.ReguAlpha = 0.01;
    end
end
bD = 1;
if ~exist('D','var') || isempty(D)
    bD = 0;
end
if isfield(options,'Kernel') && options.Kernel
    K = data;
else
    K = constructKernel(data,[],options);
end
clear data;
nSmp = size(K,1);
if size(W,1) ~= nSmp
    error('W and data mismatch!');
end
if bD && (size(D,1) ~= nSmp)
    error('D and data mismatch!');
end

% K_orig = K;
sumK = sum(K,2);
H = repmat(sumK./nSmp,1,nSmp);
K = K - H - H' + sum(sumK)/(nSmp^2);
K = max(K,K');
clear H;
%======================================
% SVD
%======================================
if bPCA
    [eigvector_PCA, eigvalue_PCA] = eig(K);
    eigvalue_PCA = diag(eigvalue_PCA);
    
    maxEigValue = max(abs(eigvalue_PCA));
    eigIdx = find(eigvalue_PCA/maxEigValue < 1e-6);
    if length(eigIdx) < 1
        [dump,eigIdx] = min(eigvalue_PCA);
    end
    eigvalue_PCA(eigIdx) = [];
    eigvector_PCA(:,eigIdx) = [];
    K = eigvector_PCA;
    clear eigvector_PCA
    
    if bD
        DPrime = K*D*K;
        DPrime = max(DPrime,DPrime');
    end
else
    if bD
        DPrime = K*D*K;
    else
        DPrime = K*K;
    end
    for i=1:size(DPrime,1)
        DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
    end
    DPrime = max(DPrime,DPrime');
end
WPrime = K*W*K;
WPrime = max(WPrime,WPrime');
%======================================
% Generalized Eigen
%======================================
dimMatrix = size(WPrime,2);
if ReducedDim > dimMatrix
    ReducedDim = dimMatrix;
end
if isfield(options,'bEigs')
    bEigs = options.bEigs;
else
    if (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        bEigs = 1;
    else
        bEigs = 0;
    end
end
if bEigs
    %disp('use eigs to speed up!');
    option = struct('disp',0);
    if bPCA && ~bD
        [eigvector, eigvalue] = eigs(WPrime,ReducedDim,'la',option);
    else
        [eigvector, eigvalue] = eigs(WPrime,DPrime,ReducedDim,'la',option);
    end
    eigvalue = diag(eigvalue);
else
    if bPCA && ~bD
        [eigvector, eigvalue] = eig(WPrime);
    else
        [eigvector, eigvalue] = eig(WPrime,DPrime);
    end
    eigvalue = diag(eigvalue);
    
    [dump, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:,index);
    
    if ReducedDim < size(eigvector,2)
        eigvector = eigvector(:, 1:ReducedDim);
        eigvalue = eigvalue(1:ReducedDim);
    end
end
if bPCA
    eigvalue_PCA = eigvalue_PCA.^-1;
    eigvector = K*(repmat(eigvalue_PCA,1,length(eigvalue)).*eigvector);
end
% tmpNorm = sqrt(sum((eigvector'*K_orig).*eigvector',2));
tmpNorm = sqrt(sum((eigvector'*K).*eigvector',2));
eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1);
end

function W = constructW(fea,options)
%	Usage:
%	W = constructW(fea,options)
%	fea: Rows of vectors of data points. Each row is x_i
%   options: Struct value in Matlab. The fields in options that can be set:
%           NeighborMode -  Indicates how to construct the graph.
%                                        Choices are[Default 'KNN']:
%                          'KNN'  -  k = 0 Complete graph
%                                         k > 0
%                                         Put an edge between two nodes if and
%                                         only if they are among the k nearst
%                                         neighbors of each other. You are
%                                         required to provide the parameter k in
%                                         the options. Default k=5.
%                 'Supervised'  -  k = 0.  Put an edge between two nodes if and only if they belong to same class.
%                                          k > 0.  Put an edge between two nodes if they belong to same class and they
%                                                     are among the k nearst neighbors of each other.  Default: k=0
%                                          You are required to provide the label information gnd in the options.
%           WeightMode   -  Indicates how to assign weights for each edge
%                                       in the graph. Choices are:
%                       'Binary' - 0-1 weighting. Every edge receiveds weight of 1.
%               'HeatKernel' - If nodes i and j are connected, put weight
%                                        W_ij = exp(-norm(x_i - x_j)/2t^2). You are
%                                        required to provide the parameter t. [Default One]
%                      'Cosine' - If nodes i and j are connected, put weight cosine(x_i,x_j).
%                              k    -   The parameter needed under 'KNN' NeighborMode. Default will be 5.
%                        gnd     -   The parameter needed under 'Supervised' NeighborMode.
%                                        Colunm vector of the label information for each data point.
%                     bLDA      -   0 or 1. Only effective under 'Supervised' NeighborMode.
%                                        If 1, the graph will be constructed to make LPP exactly same as LDA.
%                                        Default will be 0.
%                              t     -    The parameter needed under 'HeatKernel' WeightMode. Default will be 1
%   bNormalized   -   0 or 1. Only effective under 'Cosine' WeightMode.
%                              Indicates whether the fea are already be normalized to 1. Default will be 0
%      bSelfConnected  -   0 or 1. Indicates whether W(i,i) == 1. Default 0
%                                      if 'Supervised' NeighborMode & bLDA == 1,
%                                      bSelfConnected will always be 1. Default 0.
%            bTrueKNN  -   0 or 1. If 1, will construct a truly kNN graph (Not symmetric!).
%                                   Default will be 0. Only valid for 'KNN' NeighborMode
%    options.NeighborMode   =   'KNN' or 'Supervised'
%    options.k   -   needed under 'KNN' NeighborMode. Default will be 5.
%    options.gnd   -   needed under 'Supervised' NeighborMode.
%    options.WeightMode  =  'Binary' or 'HeatKernel' or 'Cosine'
%    options.t   -   needed under 'HeatKernel' WeightMode. Default will be 1
%    options.bLDA   -   Only effective under 'Supervised' NeighborMode.
%    options.bNormalized   -   Only effective under 'Cosine' WeightMode.Default will be 0.
%                                             Indicates whether the fea are already be normalized to 1.
%    options.bSelfConnected   -   Indicates whether W(i,i) == 1. Default 0. if 'Supervised' NeighborMode & bLDA == 1,
%                                                  bSelfConnected will always be 1. Default 0.
%    options.bTrueKNN   -   Only valid for 'KNN' NeighborMode. If 1, will construct a truly kNN graph (Not symmetric!).
%                                         Default will be 0.
%    Examples:
%       fea = rand(50,15);
%       options = [];
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 1;
%       W = constructW(fea,options);
%       fea = rand(50,15);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.WeightMode = 'HeatKernel';
%       options.t = 1;
%       W = constructW(fea,options);
%       fea = rand(50,15);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.bLDA = 1;
%       W = constructW(fea,options);
%    For more details about the different ways to construct the W, please
%    refer:
%       Deng Cai, Xiaofei He and Jiawei Han, "Document Clustering Using
%       Locality Preserving Indexing" IEEE TKDE, Dec. 2005.
%    Written by Deng Cai (dengcai2 AT cs.uiuc.edu), April/2004, Feb/2006,May/2007
bSpeed  = 1;

if (~exist('options','var'))
    options = [];
end

if isfield(options,'Metric')
    warning('This function has been changed and the Metric is no longer be supported');
end

if ~isfield(options,'bNormalized')
    options.bNormalized = 0;
end
%=================================================
if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

switch lower(options.NeighborMode)
    case {lower('KNN')}  %For simplicity, we include the data point itself in the kNN
        if ~isfield(options,'k')
            options.k = 5;
        end
    case {lower('Supervised')}
        if ~isfield(options,'bLDA')
            options.bLDA = 0;
        end
        if options.bLDA
            options.bSelfConnected = 1;
        end
        if ~isfield(options,'k')
            options.k = 0;
        end
        if ~isfield(options,'gnd')
            error('Label(gnd) should be provided under ''Supervised'' NeighborMode!');
        end
        if ~isempty(fea) && length(options.gnd) ~= size(fea,1)
            error('gnd doesn''t match with fea!');
        end
    otherwise
        error('NeighborMode does not exist!');
end
%=================================================
if ~isfield(options,'WeightMode')
    options.WeightMode = 'HeatKernel';
end

bBinary = 0;
bCosine = 0;
switch lower(options.WeightMode)
    case {lower('Binary')}
        bBinary = 1;
    case {lower('HeatKernel')}
        if ~isfield(options,'t')
            nSmp = size(fea,1);
            if nSmp > 3000
                D = EuDist2(fea(randsample(nSmp,3000),:));
            else
                D = EuDist2(fea);
            end
            options.t = mean(mean(D));
        end
    case {lower('Cosine')}
        bCosine = 1;
    otherwise
        error('WeightMode does not exist!');
end
%=================================================
if ~isfield(options,'bSelfConnected')
    options.bSelfConnected = 0;
end
%=================================================
if isfield(options,'gnd')
    nSmp = length(options.gnd);
else
    nSmp = size(fea,1);
end
maxM = 62500000; %500M
BlockSize = floor(maxM/(nSmp*3));
if strcmpi(options.NeighborMode,'Supervised')
    Label = unique(options.gnd);
    nLabel = length(Label);
    if options.bLDA
        G = zeros(nSmp,nSmp);
        for idx=1:nLabel
            classIdx = options.gnd==Label(idx);
            G(classIdx,classIdx) = 1/sum(classIdx);
        end
        W = sparse(G);
        return;
    end
    
    switch lower(options.WeightMode)
        case {lower('Binary')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); % sort each row
                    clear D dump;
                    idx = idx(:,1:options.k+1);
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = 1;
                    idNow = idNow+nSmpClass;
                    clear idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
                G = max(G,G');
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = 1;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end
            
            W = sparse(G);
        case {lower('HeatKernel')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); % sort each row
                    clear D;
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                    dump = exp(-dump/(2*options.t^2));
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = dump(:);
                    idNow = idNow+nSmpClass;
                    clear dump idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    D = exp(-D/(2*options.t^2));
                    G(classIdx,classIdx) = D;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end
            
            W = sparse(max(G,G'));
        case {lower('Cosine')}
            if ~options.bNormalized
                fea = NormalizeFea(fea);
            end
            
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = fea(classIdx,:)*fea(classIdx,:)';
                    [dump idx] = sort(-D,2); % sort each row
                    clear D;
                    idx = idx(:,1:options.k+1);
                    dump = -dump(:,1:options.k+1);
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = dump(:);
                    idNow = idNow+nSmpClass;
                    clear dump idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = fea(classIdx,:)*fea(classIdx,:)';
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end
            
            W = sparse(max(G,G'));
        otherwise
            error('WeightMode does not exist!');
    end
    return;
end
if bCosine && ~options.bNormalized
    Normfea = NormalizeFea(fea);
end
if strcmpi(options.NeighborMode,'KNN') && (options.k > 0)
    if ~(bCosine && options.bNormalized)
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = EuDist2(fea(smpIdx,:),fea,0);
                
                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = min(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 1e100;
                    end
                else
                    [dump idx] = sort(dist,2); % sort each row
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                end
                
                if ~bBinary
                    if bCosine
                        dist = Normfea(smpIdx,:)*Normfea';
                        dist = full(dist);
                        linidx = [1:size(idx,1)]';
                        dump = dist(sub2ind(size(dist),linidx(:,ones(1,size(idx,2))),idx));
                    else
                        dump = exp(-dump/(2*options.t^2));
                    end
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = 1;
                end
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
                
                dist = EuDist2(fea(smpIdx,:),fea,0);
                
                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = min(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 1e100;
                    end
                else
                    [dump idx] = sort(dist,2); % sort each row
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                end
                
                if ~bBinary
                    if bCosine
                        dist = Normfea(smpIdx,:)*Normfea';
                        dist = full(dist);
                        linidx = [1:size(idx,1)]';
                        dump = dist(sub2ind(size(dist),linidx(:,ones(1,size(idx,2))),idx));
                    else
                        dump = exp(-dump/(2*options.t^2));
                    end
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = 1;
                end
            end
        end
        
        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    else
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);
                
                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = max(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 0;
                    end
                else
                    [dump idx] = sort(-dist,2); % sort each row
                    idx = idx(:,1:options.k+1);
                    dump = -dump(:,1:options.k+1);
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);
                
                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = max(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 0;
                    end
                else
                    [dump idx] = sort(-dist,2); % sort each row
                    idx = idx(:,1:options.k+1);
                    dump = -dump(:,1:options.k+1);
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
            end
        end
        
        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    end
    
    if bBinary
        W(logical(W)) = 1;
    end
    
    if isfield(options,'bSemiSupervised') && options.bSemiSupervised
        tmpgnd = options.gnd(options.semiSplit);
        
        Label = unique(tmpgnd);
        nLabel = length(Label);
        G = zeros(sum(options.semiSplit),sum(options.semiSplit));
        for idx=1:nLabel
            classIdx = tmpgnd==Label(idx);
            G(classIdx,classIdx) = 1;
        end
        Wsup = sparse(G);
        if ~isfield(options,'SameCategoryWeight')
            options.SameCategoryWeight = 1;
        end
        W(options.semiSplit,options.semiSplit) = (Wsup>0)*options.SameCategoryWeight;
    end
    
    if ~options.bSelfConnected
        W = W - diag(diag(W));
    end
    
    if isfield(options,'bTrueKNN') && options.bTrueKNN
        
    else
        W = max(W,W');
    end
    
    return;
end
% strcmpi(options.NeighborMode,'KNN') & (options.k == 0)
% Complete Graph
switch lower(options.WeightMode)
    case {lower('Binary')}
        error('Binary weight can not be used for complete graph!');
    case {lower('HeatKernel')}
        W = EuDist2(fea,[],0);
        W = exp(-W/(2*options.t^2));
    case {lower('Cosine')}
        W = full(Normfea*Normfea');
    otherwise
        error('WeightMode does not exist!');
end

if ~options.bSelfConnected
    for i=1:size(W,1)
        W(i,i) = 0;
    end
end
W = max(W,W');
end
function [Ly,Lx] = TestLambda(Ky,Kx,Ktg,Ktp,Ay,Ax,para)
t = Ky'*Ay;
t0 = t*t';
Imat = eye(size(t0,1));
Sy = t0+(para.lambda+para.tao)*Imat;
Qy = t*Ay'*Ktg;
t1 = Kx'*Ax;
t2 = t1*t1';
Imat1 = eye(size(t2,1));
Sx = t2+(para.lambda+para.tao)*Imat1;
Qx = t1*Ax'*Ktp;
SxSy = Sx*Sy;
SySx = Sy*Sx;
temp = inv(SxSy+para.theta*eye(size(SxSy,1)));
temp1 = inv(SySx+para.theta*eye(size(SySx,1)));
Imat2 = eye(size(temp,1));
Imat3 = eye(size(temp1,1));
Uy = Imat2-para.tao^2*temp;
Ux = Imat3-para.tao^2*temp1;
Ly = ComLambda(Uy,Sy,Sx,Qy,Qx,para);
Lx = ComLambda(Ux,Sx,Sy,Qx,Qy,para);
end
function L = ComLambda(U1,S1,S2,Q1,Q2,para)
S1U1 = S1*U1;
temp = (S1U1+para.theta*eye(size(S1U1,1)))\Q1;
S2S1U1 = S2*S1U1;
temp1 = (S2S1U1+para.theta*eye(size(S2S1U1,1)))\Q2;
L = temp+para.tao*temp1;
end