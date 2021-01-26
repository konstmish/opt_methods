# Optimization methods
### Benchmarking optimization methods on convex problems.
## Structure
### First order
Gradient Descent (GD), Polyak's Heavy-ball, Incremental Gradient (IG), Mirror Descent (MD), Nesterov's acceleration (Nesterov), Nesterov with restarts (RestNest).

Adaptive: AdaGrad, Adaptive GD (AdGD), Accelerated AdGD (AdgdAccel), Polyak.
### Second order
Newton.

Stochastic: Stochastic Newton, Stochastic Cubic Regularization.

Qausi-Newotn: BFGS, DFP, L-BFGS, Shor, SR1.
### Stochastic first order
SGD, Root-SGD, Stochastic Variance Reduced Gradient (SVRG), Random Reshuffling (RR).
### Notebooks
Examples of running the methods on convex problems: linear regression (to appear), logistic regression, entropy minimization (to appear).

Benchmarking wall-clock time of some numpy and scipy operations to show how losses should be implemented.

