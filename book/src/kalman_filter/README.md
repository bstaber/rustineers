## Model formulation

We consider the following general state space model made of a state equation and an observation equation:

$$
x_k = \Theta x_{k-1} + W_k, \quad W_k \sim \mathcal{N}(0, R)
$$
$$
y_k = A_k x_k + V_k, \quad V_k \sim \mathcal{N}(0, S)
$$

where:

- $x_k \in \mathbb{R}^n$ is the state vector at time step $k$,
- $y_k \in \mathbb{R}^p$ is the observation/measurement vector.

We also have the following hyperparameters assumed to be known:
- $\Theta \in \mathbb{R}^{n \times n}$ is the state transition matrix,
- $A_k \in \mathbb{R}^{p \times n}$ is the observation matrix,
- $R \in \mathbb{R}^{n \times n}$ is the process noise covariance,
- $S \in \mathbb{R}^{p \times p}$ is the observation noise covariance.

The aim is to estimate the state $x_k$ given the noisy observations $y_k, y_{k-1}, \dots, y_1$.

## Kalman filtering

We start with an initial state $x_0 \sim \mathcal{N}(x_0^0, P_0^0)$ and we want to determine the conditional distribution $p(x_1 | y_1)$ of the next state. Using Baye's rule, one has
$$
p(x_1 | y_1) \propto p(y_1 | x_1) p(x_1)\,.
$$
The likelihood $p(y_1 | x_1)$ can easily be determined thanks to the chosen observation equation. The marginal distribution $p(x_1)$ can be obtained by marginalizing $p(x_1, x_0)$.
This marginal distribution can be seen as our prior knowledge before seeing $y_1$. It can easily be shown that
$$
p(x_1) \propto \mathcal{N}(x_1^0, P_1^0)\,,
$$
with
$$
x_1^0 = \Theta x_0^0\,, \quad P_1^0 = \Theta P_0^0 \Theta^T + R\,.
$$
This first estimate is what we will call the **prediction step**. Using this result, we can determine $p(x_1 | y_1)$:
$$
p(x_1 | y_1) \propto \mathcal{N}(x_1^1, P_1^1)\,,
$$
with
$$
x_1^1 = x_1^0 + K_1(y_1 - A_1 x_1^0)\,, \quad P_1^1 = (I - K_1 A_1)P_1^0\,,
$$
which gives us our estimate the next state given the observation $y_1$. This is what we call the **update step**.

## General equations

For an arbitrary time step $k$, the **prediction step** yields:
$$
x_t^{t-1} = \Theta x_{t-1}^{t-1}\,, \quad P_t^{t-1} = \Theta P_{t-1}^{t-1} \Theta^T + R\,,
$$
and the **update step** is given by
$$
x_t^t = x_t^{t-1} + K_t(y_t - A_t x_t^{t-1})\,, \quad P_t^t = (I - K_t A_t)P_t^{t-1}\,,
$$
where the Kalman gain $K_t$ is defined as
$$
K_t = P_t^{t-1} A_t^T(A_t P_t^{t-1}A_t^T + S)^{-1}\,.
$$

Implementing the Kalman filter boils down to implement these few equations!