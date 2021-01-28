import copy
import numpy as np

from optimizer import Optimizer


class AdgdAccel(Optimizer):
    """
    Accelerated gradient descent with adaptive stepsize and momentum estimation
    using local values of smoothness (gradient Lipschitzness) and strong convexity.
    Momentum is used as given by Nesterov's acceleration.
    
    Arguments:
        lr0 (float, optional): a small value that idealy should be smaller than the
            inverse (local) smoothness constant. Does not affect performance too much.
    """
    def __init__(self, lr0=1e-6, *args, **kwargs):
        super(AdgdAccel, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        
    def step(self):
        self.grad = self.loss.gradient(self.x_nest)
        self.estimate_new_stepsize()
        self.estimate_new_momentum()
        self.x_nest_old = copy.deepcopy(self.x_nest)
        self.x_old = copy.deepcopy(self.x)
        self.grad_old = copy.deepcopy(self.grad)
        self.x = self.x_nest - self.lr*self.grad
        self.x_nest = self.x + self.momentum*(self.x - self.x_old)
        
    def estimate_new_stepsize(self):
        if self.grad_old is not None:
            self.L = self.loss.norm(self.grad-self.grad_old) / self.loss.norm(self.x_nest-self.x_nest_old)
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
        self.x_nest = copy.deepcopy(self.x)
        self.momentum = 0
        self.lr = self.lr0
        self.mu = 1 / self.lr
        self.trace.lrs = [self.lr]
        self.trace.momentums = [0]
        self.theta = np.inf
        self.theta_mu = 1
        self.grad_old = None
        
    def update_trace(self):
        super(AdgdAccel, self).update_trace()
        self.trace.lrs.append(self.lr)
        self.trace.momentums.append(self.momentum)
