function Yhat = predict_lstm(model, Xseq)
% Generates predictions for a dataset of multiple sequences: iterates over a batch of variable-length sequences, 
% passing each through the LSTM forward propagation pass to compute the final network predictions.
% Inputs: model: Struct containing the trained LSTM weights, biases, and parameters.
%  Xseq: Mx1 cell array, where each cell contains a sequence matrix of size [feature_dim (d) x time_steps (T)]
% Outputs: Yhat: Mx1 vector containing the scalar predictions for all M sequences. 
    M = numel(Xseq); % number of sequences 
    Yhat = zeros(M,1);
    for i = 1:M
        Yhat(i) = forward_one(model, Xseq{i}); % evaluate sequence with forward pass
    end
end