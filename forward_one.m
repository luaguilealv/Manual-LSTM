function yhat = forward_one(model, X)
% forward propagation pass of LSTM recurrent neural network cell for a single input sequence
% processes input sequence sequentially over time
% computes the internal gate activations, updates the cell memory state, and computes final network output.

% Inputs: model: Struct with weights (W, U), biases (b), and hidden size (H). X: Input data matrix of size [feature_dim (d), time_steps (T)]
% Outputs: yhat: Predicted output vector at the final time step

% Initialize 
    H = model.H;
    T = size(X,2);
    h = zeros(H,1);
    c = zeros(H,1);
    for t = 1:T
        xt = X(:,t);
        ft = sigmoid(model.Wf*xt + model.Uf*h + model.bf);
        it = sigmoid(model.Wi*xt + model.Ui*h + model.bi);
        gt = tanh(   model.Wg*xt + model.Ug*h + model.bg);
        ot = sigmoid(model.Wo*xt + model.Uo*h + model.bo);
        
        % Update Cell State and Hidden State
        c = ft.*c + it.*gt;
        h = ot.*tanh(c);
    end
    yhat = model.Wy*h + model.by; % Compute Final Output Linear Layer (final regresion layer)
end