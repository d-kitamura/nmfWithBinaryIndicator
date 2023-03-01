% Nonnegative matrix factorization with binary indicator matrix
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%

clear; close all; clc;

%% Parameters
% Set simulation data parameters
seed = 1; % pseudo-random seed
nRow = 100; % number of rows in X
nCol = 100; % number of columns in X
nBasis = 10; % number of bases in X (nBasis=rank(trueX))
missingRate = 0.5; % rate of missing values included in observed matrix

% Set NMF parameters
nIter = 1000000; % number of iterations in NMF optimization
debugFlag = true; % whether plot convergence curve (true/false)

%% Main processes
% Produce simulation data
rng(seed); % Fix pseudo-random stream for reproducibility
trueW = rand(nRow, nBasis); % true basis matrix
trueH = rand(nBasis, nCol); % true activation matrix
trueX = trueW * trueH; % observed nonnegative matrix (rank(trueX) = nBases)
B = (rand(nRow, nCol) > missingRate); % binary indicator matrix (0 means missing value, and 1 means valid elements)
obsX = B .* trueX; % observed nonnegative matrix with missing values (missing values are set to 0, not NaN)

% Do NMF with binary indicator matrix
[estW, estH, estX, costNmf] = calcNmfBinaryInd(obsX, B, nBasis, nIter, debugFlag);

%% Results
% Show convergence curve
if debugFlag
    figure; semilogy((0:nIter), costNmf); grid on;
    xlabel("Number of iterations"); ylabel("Cost function value in NMF");
    title("Convergence behavior");
end

% Show true, observed, and recovered matrices
figure("Position", [100, 100, 1200, 400]);
tileObj = tiledlayout(1, 3, "TileSpacing", "compact", "Padding", "compact");
ax1 = nexttile([1, 1]); hdl1 = imagesc(trueX); title("True matrix (trueX)");
ax2 = nexttile([1, 1]); hdl2 = imagesc(obsX); title("Observed matrix with missing values (obsX)");
ax3 = nexttile([1, 1]); hdl3 = imagesc(estX); title("Recovered matrix (estX)");
set(hdl2, "AlphaData", ~(obsX == 0)); % set missing values to transparent color
maxVal = max(trueX, [], "all"); minVal = min(trueX, [], "all"); % get maximum and minimum values in trueX
clim(ax1, [minVal, maxVal]); % set color range
clim(ax2, [minVal, maxVal]); % set color range
clim(ax3, [minVal, maxVal]); % set color range
colorbar; % show color bar
linkaxes([ax1, ax2, ax3], "xy"); % link xy axes of three images

% Show recovering error
err = norm(trueX - estX, "fro")^2; % squared Euclidean distance
fprintf("Approximation error: " + 10*log10(err) + " [dB]\n");