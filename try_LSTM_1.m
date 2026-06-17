%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CODE DEVELOPED BY LUCIA 

% Try manual neural network training. I have a neural nework with an Input
% layer + LSTM layer + regresion output layer
% Optimizer is Stochastic gradient descent
% The process of forward pass and BPTT is writen manually- no build in
% functions 
% Work based in: @misc{higham2018deeplearningintroductionapplied,
%       title={Deep Learning: An Introduction for Applied Mathematicians}, 
%       author={Catherine F. Higham and Desmond J. Higham},
%       year={2018},
%       eprint={1801.05894},
%       archivePrefix={arXiv},
%       primaryClass={math.HO},
%       url={https://arxiv.org/abs/1801.05894}, 
% }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;
% generated data
%% Create time series 
rng(1);
N = 600;   % time steps
d = 5;     % features
X = randn(N,d);

% Make y depend on history so LSTM is meaningful
y = zeros(N,1);
for t = 3:N
    y(t) = 0.8*y(t-1) - 0.2*y(t-2) + 0.6*sin(X(t,1)) - 0.3*X(t,3) + 0.1*randn();
end

%% Build sequences using sliding windows 
T = 20;                 % sequence length

[Xseq, Yseq] = makeSequences(X, y, T); % Xseq cells: [d x T]

%% Train/test split
rng(1);
M = numel(Xseq);
idx = randperm(M);
ntrain = round(0.8*M);
tr = idx(1:ntrain);
te = idx(ntrain+1:end);

Xtr = Xseq(tr);  Ytr = Yseq(tr);
Xte = Xseq(te);  Yte = Yseq(te);

%% LSTM hyperparameters 
H = 50;             % hidden units
eta = 0.001;          % learning rate
Niter = 200000;      % SGD steps
evalEvery = 2000;    % cost evaluation interval (to not have the cost for every iteration)

%% Train manual LSTM
tic;
model = my_lstm(Xtr, Ytr, d, H, eta, Niter, evalEvery);
toc;

%% Evaluate 
Yhat_tr = predict_lstm(model, Xtr);
Yhat_te = predict_lstm(model, Xte);

mse_tr = mean((Yhat_tr - Ytr).^2);
mse_te = mean((Yhat_te - Yte).^2);

fprintf('\nFinal performance\n');
fprintf('Train MSE: %.6g\n', mse_tr);
fprintf('Test  MSE: %.6g\n', mse_te);

%% Plots
figure; semilogy(model.costHistory); grid on;
xlabel('Evaluation step'); ylabel('Train MSE'); title('Training curve (manual LSTM)');

figure;
subplot(1,2,1);
scatter(Ytr, Yhat_tr, 'filled'); grid on; xlabel('True'); ylabel('Pred'); title('Train');

subplot(1,2,2);
scatter(Yte, Yhat_te, 'filled'); grid on; xlabel('True'); ylabel('Pred'); title('Test');

figure;
histogram(Yhat_te - Yte, 20); grid on;
xlabel('Residual (yhat - y)'); title('Test residuals');

%% Data sequences
function [Xseq, Yseq] = makeSequences(X, y, T)

    X = double(X);
    y = double(y);

    [N,d] = size(X);

    M = N - T;   % number of sequences

    Xseq = cell(M,1);
    Yseq = zeros(M,1);

    for i = 1:M
        Xseq{i} = X(i:i+T-1, :)';   % [d x T]
        Yseq(i) = y(i+T);           % next-step prediction
    end
end