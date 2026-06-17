# Manual-LSTM

A transparent, from-scratch implementation of a Long Short-Term Memory (LSTM) recurrent neural network in MATLAB. This repository bypasses high-level deep learning wrappers (such as `trainnet` or `trainNetwork`) to program the core mathematical mechanisms explicitly—including initialization, manual gate activation loops, forward propagation, and Backpropagation Through Time (BPTT).

## Mathematical Foundation & References

The algorithmic framework and matrix-based indexing conventions utilized in this code are directly adapted from:

* **Primary Reference:** Higham, C. F., & Higham, D. J. (2018). *Deep Learning: An Introduction for Applied Mathematicians*. arXiv preprint arXiv:1801.05894. [Available here](https://arxiv.org/abs/1801.05894).
* **BPTT Gradient Calculus Reference:** [Christina Kouridi - Backpropagation in an LSTM cell](https://christinakouridi.github.io/posts/backprop-lstm/).

## Overview

The goal is to provide a clear, mathematically rigorous look into the inner workings of sequential deep learning architectures. By establishing manual matrix operations, it eliminates automated differentiation abstraction to explicitly trace how gradients and states evolve across variable-length temporal sequences.

## Architecture & File Breakdown

The code combines a self-contained environment processing time-series sequences through the following functions:

### 1. Main Pipeline & Training Core
* **`try_LSTM_1.m` Execution Flow:** Generates a synthetic nonlinear time-series dataset where the target variable $y_t$ depends on past temporal history. It splits the data into training (80%) and testing (20%) subsets using sliding windows, kicks off the custom engine, and generates performance diagnostic plots (residual analysis, truth-vs-prediction tracking, and logarithmic MSE convergence scales).
* **`my_lstm.m`:** The primary optimization core. Initializes all gate weights and biases randomly, manages the overarching SGD iteration loops, provisions sequence indexing, tracks cell state memory history, runs backpropagation calculus over unrolled time steps, and periodically evaluates mean squared errors (MSE).

### 2. Analytical Subroutines
* **`forward_one.m`:** Computes the isolated sequential forward propagation pass for an individual input sequence matrix, step-by-step updating structural activations.
* **`predict_lstm.m`:** Unpacks high-level batch cell data vectors and evaluates them across `forward_one` routines to output a collective matrix of scalar regression predictions.
* **`makeSequences.m`:** Data pre-processing utility transforming raw multi-feature data matrices into a collection of sliding windows formatted into cell structures of shape `[d x T]`.
* **`sigmoid.m`:** Base mathematical implementation of the gate activation function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

## Algorithmic Framework

### 1. Forward Pass Details
For each sequence, the network tracks hidden state $h_t \in \mathbb{R}^H$ and cell memory state $c_t \in \mathbb{R}^H$. At each time step $t \in [1, T]$, given an input slice $x_t \in \mathbb{R}^d$, the four core internal gating layers compute:

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{(Forget Gate)}$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad \text{(Input Gate)}$$
$$g_t = \tanh(W_g x_t + U_g h_{t-1} + b_g) \quad \text{(Candidate Memory Vector)}$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad \text{(Output Gate)}$$

**State Evolution Mapping:**
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
$$h_t = o_t \odot \tanh(c_t)$$
 
The final sequence output is passed to a linear regression layer evaluated at the terminal time step $T$:
$$\hat{y} = W_y h_T + b_y$$

### 2. Backpropagation Through Time (BPTT)
The system minimizes a mean squared error objective function per instance: $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$. The gradient updates are calculated by unraveling temporal sequences backwards ($t = T \rightarrow 1$), accumulating partial error matrices manually:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = (\hat{y} - y)$$
$$\delta o_t = \delta h_t \odot \tanh(c_t) \odot o_t \odot (1 - o_t)$$
$$\delta c_t = \delta c_{\text{next}} \odot f_{\text{next}} + \delta h_t \odot o_t \odot (1 - \tanh^2(c_t))$$

Gradients are accumulated across all time horizons for parameters ($W$, $U$, $b$) before parameter scaling is performed via Stochastic Gradient Descent:
$$\theta = \theta - \eta \cdot d\theta$$

## Hyperparameters and Settings

The standard tracking configuration deployed in the routine contains:
* **Sequence Length ($T$):** 20 time steps
* **Input Features ($d$):** 5 dimensions
* **Hidden Size ($H$):** 50 hidden units
* **Learning Rate ($\eta$):** 0.001
* **Optimization Iterations ($N_{\text{iter}}$):** 200,000 SGD steps
* **Evaluation Frequency ($evalEvery$):** 2,000 steps (saves memory and minimizes runtime computational drag)

## Getting Started

Simply open base MATLAB, place all functions in your active working directory, and run the main training script file.

```matlab
% Run the execution script directly in the MATLAB console
run('try_LSTM_1.m')
