function model = my_lstm(Xseq, Yseq, d, H, eta, Niter, evalEvery)
% Manual single-layer LSTM, sequence-to-one regression
% Xseq{i}: [d x T], Yseq(i): scalar
% Loss per sample: 0.5*(yhat - y)^2
% Output: yhat = Wy*h_T + by (linear)
% H = number of hidden units in the LSTM, hidden state h_t \in R^H
% W \in R^Hxd, b \in R^H, U \in R^HxH


    M = numel(Xseq);

    % Initialize parameters: LSTM parameters for the 4 gates
    rng(5000);
    % Gates: f, i, g (c tilda in notes, candidate), o
    Wf = 0.5*randn(H,d); Uf = 0.5*randn(H,H); bf = 0.5*randn(H,1);
    Wi = 0.5*randn(H,d); Ui = 0.5*randn(H,H); bi = 0.5*randn(H,1);
    Wg = 0.5*randn(H,d); Ug = 0.5*randn(H,H); bg = 0.5*randn(H,1);
    Wo = 0.5*randn(H,d); Uo = 0.5*randn(H,H); bo = 0.5*randn(H,1);

    % Output layer: regression output layer, outputs must be any real number, and assume linear output
    Wy = 0.5*randn(1,H);
    by = 0.5*randn(1,1);

    % Cost history (evaluated every evalEvery)
    nEvals = ceil(Niter / evalEvery);
    costHistory = zeros(nEvals,1);
    evalCount = 0;

    for it = 1:Niter
        k = randi(M);
        X = Xseq{k};      % [d x T]
        y = Yseq(k);      % scalar
        T = size(X,2);

        % Forward pass: following the steps in the gates (normal computations)
        % cell structure to save infromation: look code for ANN (explained)
        h = cell(T,1);
        c = cell(T,1);
        f = cell(T,1);
        ii = cell(T,1);   
        g = cell(T,1);   % c tilda in notes
        o = cell(T,1);

        zf = cell(T,1); zi = cell(T,1); zg = cell(T,1); zo = cell(T,1);
        % save info
        hprev = zeros(H,1);
        cprev = zeros(H,1);

        for t = 1:T
            xt = X(:,t);
            % gates calculations
            zf{t} = Wf*xt + Uf*hprev + bf;   f{t}  = sigmoid(zf{t});
            zi{t} = Wi*xt + Ui*hprev + bi;   ii{t} = sigmoid(zi{t});
            zg{t} = Wg*xt + Ug*hprev + bg;   g{t}  = tanh(zg{t});
            zo{t} = Wo*xt + Uo*hprev + bo;   o{t}  = sigmoid(zo{t});
            % cell update-->c, hidden state update-->h
            c{t} = f{t} .* cprev + ii{t} .* g{t};
            h{t} = o{t} .* tanh(c{t});
            % memory updates 
            hprev = h{t};
            cprev = c{t};
        end
        % output layer regression
        hT = h{T};
        yhat = Wy*hT + by;          % linear output
        % Back propagation: d are deltas
        % loss = 0.5*(yhat - y)^2, compute derivative
        dy = (yhat - y);            % derivative of 0.5*(...)^2 is (yhat - y)

        % Gradients: output layer
        dWy = dy * hT.';
        dby = dy;
        % derivatives that repeat
        % Backprop signal into h_T
        dh_next = Wy.' * dy;        % [H x 1]
        dc_next = zeros(H,1);

        % Initialize gate grads
        dWf = zeros(H,d); dUf = zeros(H,H); dbf = zeros(H,1);
        dWi = zeros(H,d); dUi = zeros(H,H); dbi = zeros(H,1);
        dWg = zeros(H,d); dUg = zeros(H,H); dbg = zeros(H,1);
        dWo = zeros(H,d); dUo = zeros(H,H); dbo = zeros(H,1);

        % BPTT: backpropagation through time
        % all the derivatives!!!!! --> my notes, write in latex
        for t = T:-1:1
            xt = X(:,t);

            if t == 1
                hprev = zeros(H,1);
                cprev = zeros(H,1);
            else
                hprev = h{t-1};
                cprev = c{t-1};
            end
            
            ct = c{t};
            ht = h{t};

            % h_t = o_t .* tanh(c_t)
            tanh_ct = tanh(ct);

            % dh includes contributions from output (t=T) and from future time steps
            dh = dh_next;

            % do = dh .* tanh(c_t)
            do = dh .* tanh_ct;

            % dzo = do .* sigmoid'(zo) = do .* o*(1-o)
            dzo = do .* (o{t} .* (1 - o{t}));

            % dc accumulates from dh path + future dc
            dc = dc_next + dh .* o{t} .* (1 - tanh_ct.^2);

            % c_t = f_t .* c_{t-1} + i_t .* g_t
            df = dc .* cprev;
            di = dc .* g{t};
            dg = dc .* ii{t};

            % pre-activation grads
            dzf = df .* (f{t} .* (1 - f{t}));   % sigmoid'
            dzi = di .* (ii{t} .* (1 - ii{t})); % sigmoid'
            dzg = dg .* (1 - g{t}.^2);           % tanh'

            % Accumulate parameter grads --> Reference: https://christinakouridi.github.io/posts/backprop-lstm/
            % explained
            dWf = dWf + dzf * xt.';  dUf = dUf + dzf * hprev.';  dbf = dbf + dzf;
            dWi = dWi + dzi * xt.';  dUi = dUi + dzi * hprev.';  dbi = dbi + dzi;
            dWg = dWg + dzg * xt.';  dUg = dUg + dzg * hprev.';  dbg = dbg + dzg;
            dWo = dWo + dzo * xt.';  dUo = dUo + dzo * hprev.';  dbo = dbo + dzo;

            % Backprop into previous h and c
            dh_prev = Uf.'*dzf + Ui.'*dzi + Ug.'*dzg + Uo.'*dzo;
            dc_prev = dc .* f{t};

            dh_next = dh_prev;
            dc_next = dc_prev;
        end

        % SGD update: for gates parameters and output layer
        Wf = Wf - eta*dWf; Uf = Uf - eta*dUf; bf = bf - eta*dbf;
        Wi = Wi - eta*dWi; Ui = Ui - eta*dUi; bi = bi - eta*dbi;
        Wg = Wg - eta*dWg; Ug = Ug - eta*dUg; bg = bg - eta*dbg;
        Wo = Wo - eta*dWo; Uo = Uo - eta*dUo; bo = bo - eta*dbo;

        Wy = Wy - eta*dWy;
        by = by - eta*dby;

        % Monitor training
        if mod(it, evalEvery) == 0
            evalCount = evalCount + 1;
            costHistory(evalCount) = full_mse();
            % fprintf('Iter %d/%d | Loss function: %.6g\n', it, Niter, costHistory(evalCount)); 
            % up: can be useful to print to see progress, but takes memory and takes time
        end
    end

    % Pack model, save info
    model.Wf=Wf; model.Uf=Uf; model.bf=bf;
    model.Wi=Wi; model.Ui=Ui; model.bi=bi;
    model.Wg=Wg; model.Ug=Ug; model.bg=bg;
    model.Wo=Wo; model.Uo=Uo; model.bo=bo;
    model.Wy=Wy; model.by=by;

    model.d = d;
    model.H = H;
    model.costHistory = costHistory(1:evalCount);
    model.evalEvery = evalEvery;

    function costval = full_mse() % computation of MSE
    % Computation of the 
        costvect = zeros(M,1);
        % M = number of training examples
        
        for kk = 1:M
            Xk = Xseq{kk};
        
            h = zeros(H,1);
            c = zeros(H,1);
        
            for t = 1:size(Xk,2)
                xt = Xk(:,t);
        
                ft = sigmoid(Wf*xt + Uf*h + bf);
                it = sigmoid(Wi*xt + Ui*h + bi);
                gt = tanh(   Wg*xt + Ug*h + bg);
                ot = sigmoid(Wo*xt + Uo*h + bo);
        
                c = ft.*c + it.*gt;
                h = ot.*tanh(c);
            end
        
            costvect(kk) = Wy*h + by;
        end
        
        r = costvect - Yseq;
        costval = mean(r.^2);
    end

end




