import numpy as np

from optimizer import Optimizer


class Adgd(Optimizer):
    """
    Gradient descent with adaptive stepsize estimation
    using local values of smoothness (gradient Lipschitzness).
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=1e-10, *args, **kwargs):
        super(Adgd, self).__init__(*args, **kwargs)
        self.lr = lr
        self.theta = np.inf
        self.grad_old = None
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.estimate_new_stepsize()
        self.x_old = self.x.copy()
        self.grad_old = self.grad.copy()
        self.x -= self.lr * self.grad
        
    def estimate_new_stepsize(self):
        if self.grad_old is not None:
            L = self.loss.norm(self.grad-self.grad_old) / self.loss.norm(self.x-self.x_old)
            lr_new = min(np.sqrt(1+self.theta) * self.lr, 0.5/L)
            self.theta = lr_new / self.lr
            self.lr = lr_new
    
    def init_run(self, *args, **kwargs):
        super(Adgd, self).init_run(*args, **kwargs)
        self.trace.lrs = [self.lr]
        
    def update_trace(self):
        super(Adgd, self).update_trace()
        self.trace.lrs.append(self.lr)