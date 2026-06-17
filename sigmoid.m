function s = sigmoid(z)
% LSTM: type of activation given
    s = 1 ./ (1 + exp(-z));
end