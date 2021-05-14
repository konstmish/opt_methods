# Optimization methods
This is a package containing implementations of different loss functions and optimization algorithms. The main goal of this package is to have a unified and easy-to-use comparison of iteration complexities of the algorithms, so the time comparison of methods is approximate. If you are interested in finding the best implementation of a solver for your problem, you may find the [BenchOpt package](https://benchopt.github.io/index.html) more useful.
## Structure
### First order
Gradient-based algorithms:  
Gradient Descent (GD), Polyak's Heavy-ball, Incremental Gradient (IG), Mirror Descent (MD), Nesterov's acceleration (Nesterov), Nesterov with restarts (RestNest).  
Adaptive: AdaGrad, Adaptive GD (AdGD), Accelerated AdGD (AdgdAccel), Polyak.
### Second order
Algorithms that use second-order information (second derivatives) or their approximations.  
Newton.  
Stochastic: Stochastic Newton, Stochastic Cubic Regularization.  
Qausi-Newotn: BFGS, DFP, L-BFGS, Shor, SR1.
### Stochastic first order
SGD, Root-SGD, Stochastic Variance Reduced Gradient (SVRG), Random Reshuffling (RR).
### Notebooks
1. Deterministic first-order methods: GD, acceleration, adaptive algorithms.  
2. Second-order methods and quasi-Newton algorithms: Newton, Levenberg-Marquardt, BFGS, SR1, DFP.  
Examples of running the methods on convex problems: linear regression (to appear), logistic regression, entropy minimization.  
Benchmarking wall-clock time of some numpy and scipy operations to show how losses should be implemented.