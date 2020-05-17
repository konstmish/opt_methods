import numpy as np

from optimizer import Optimizer


class AdgdAccel(Optimizer):
    """
    Accelerated gradient descent with adaptive stepsize and momentum estimation
    using local values of smoothness (gradient Lipschitzness) and strong convexity.
    Momentum is used as given by Nesterov's acceleration.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=1e-10, *args, **kwargs):
        super(AdgdAccel, self).__init__(*args, **kwargs)
        self.lr = lr
        self.mu = 1 / self.lr
        self.theta = np.inf
        self.theta_mu = 1
        self.grad_old = None
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.estimate_new_stepsize()
        self.estimate_new_momentum()
        self.x_nest_old = self.x_nest.copy()
        self.x_old = self.x.copy()
        self.grad_old = self.grad.copy()
        self.x_nest = self.x - self.lr*self.grad
        self.x = self.x_nest + self.momentum*(self.x_nest-self.x_nest_old)
        
    def estimate_new_stepsize(self):
        if self.grad_old is not None:
            self.L = self.loss.norm(self.grad-self.grad_old) / self.loss.norm(self.x-self.x_old)
            lr_new = min(np.sqrt(1+0.5*self.theta) * self.lr, 0.5/self.L)
            self.theta = lr_new / self.lr
            self.lr = lr_new
            
    def estimate_new_momentum(self):
        if self.grad_old is not None:
            mu_new = min(np.sqrt(1+0.5*self.theta_mu) * self.mu, 0.5*self.L)
            self.theta_mu = mu_new / self.mu
            self.mu = mu_new
            kappa = 1 / (self.lr*self.mu)
            self.momentum = 1 - 2 / (1+np.sqrt(kappa))
    
    def init_run(self, *args, **kwargs):
        super(AdgdAccel, self).init_run(*args, **kwargs)
        self.x_nest = self.x.copy()
        self.momentum = 0
        self.trace.lrs = [self.lr]
        self.trace.momentums = [0]
        
    def update_trace(self):
        super(AdgdAccel, self).update_trace()
        self.trace.lrs.append(self.lr)
        self.trace.momentums.append(self.momentum)