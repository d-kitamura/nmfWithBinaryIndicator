function [basisMat, coefMat, modelMat, costVal] = calcNmfBinaryInd(obsMat, binMat, nBasis, nIter, isDebug)
% Nonnegative matrix factorization with binary indicator matrix
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
% 
% [Syntax]
%   [basisMat, coefMat, modelMat, costVal] = calcNmfBinaryInd(obsMat,binMat)
%   [basisMat, coefMat, modelMat, costVal] = calcNmfBinaryInd(obsMat,binMat,nBasis)
%   [basisMat, coefMat, modelMat, costVal] = calcNmfBinaryInd(obsMat,binMat,nBasis,nIter)
%   [basisMat, coefMat, modelMat, costVal] = calcNmfBinaryInd(obsMat,binMat,nBasis,nIter,isDebug)
%
% [Inputs]
%     obsMat: nonnegative observed matrix (nRow x nCol)
%     binMat: binary indicator matrix (nRow x nCol)
%     nBasis: number of bases used in NMF modeling (scalar, default: 100)
%      nIter: numebr of iterations in NMF optimization (scalar, default: 100)
%    isDebug: calculate cost function values in each iteration or not (true or false, default: false)
%
% [Outputs]
%   basisMat: estimated nonnegative basis matrix (nRow x nBasis)
%    coefMat: estimated nonnegative activation matrix (nBasis x nCol)
%   modelMat: estimated nonnegative model matrix (nRow x nCol, basisMat*coefMat)
%       cost: convergence behavior of cost function in NMF with binary indicator matrix (nIter+1 x 1)
%

% Check arguments and set default values
arguments
    obsMat (:,:) double {mustBeNonnegative}
    binMat (:,:) logical
    nBasis (1,1) double {mustBeInteger(nBasis), mustBeNonnegative(nBasis)} = 10
    nIter (1,1) double {mustBeInteger(nIter), mustBeNonnegative(nIter)} = 1000
    isDebug (1,1) logical = false
end

% Check errors
[nRow, nCol] = size(obsMat);
if (size(binMat, 1) ~= nRow) || (size(binMat, 2) ~= nCol); error("The size of binary indicator matrix does not match to that of obsMat.\n"); end

% Set initial values of NMF variables
iniBasisMat = rand(nRow, nBasis); % random values in the range (0, 1)
iniCoefMat = rand(nBasis, nCol); % random values in the range (0, 1)

% Calculate NMF optimization
[basisMat, coefMat, costVal] = local_updateParam(obsMat, binMat, iniBasisMat, iniCoefMat, nIter, isDebug);

% Calculate model matrix
modelMat = basisMat * coefMat;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Local functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------------------------------------------------------------------------
function [W, H, cost] = local_updateParam(X, B, W, H, nIter, isDebug)
% [Variables]
%   X: observed matrix
%   B: binary index matrix
%   W: basis matrix
%   H: activation matrix

% Calculate inital cost function value
cost = zeros(nIter+1, 1);
if isDebug; cost(1, 1) = local_calcCost(X, B, W, H); end

% Initialize progress
fprintf("NMF iteration:     ");

% Iterate update rules
for iIter = 1:nIter
    % Update W
    WH = W * H;
    W = W .* ((B.*X)*H.') ./ ((B.*WH)*H.'); % update rule
    W = max(W, eps); % epsilon flooring to avoid numerical instability

    % Update H
    WH = W * H;
    H = H .* (W.'*(B.*X)) ./ (W.'*(B.*WH)); % update rule
    H = max(H, eps); % epsilon flooring to avoid numerical instability

    % Calculate current cost function value
    if isDebug; cost(iIter+1, 1) = local_calcCost(X, B, W, H); end

    % Show progress
    if rem(iIter, nIter/100) == 0; fprintf("\b\b\b\b\b%4.0f%%", 100*iIter/nIter); end
end
fprintf("\n");
end

% -------------------------------------------------------------------------
function cost = local_calcCost(X, B, W, H)
cost = norm(B.*(X - W*H), "fro")^2; % squared Euclidean distance
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%