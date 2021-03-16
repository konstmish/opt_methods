import copy
import numpy as np

from optimizer import Optimizer


class Adgd(Optimizer):
    """
    Gradient descent with adaptive stepsize estimation
    using local values of smoothness (gradient Lipschitzness).
    
    Arguments:
        lr0 (float, optional): a small value that idealy should be smaller than the
            inverse (local) smoothness constant. Does not affect performance too much.
    """
    def __init__(self, lr0=1e-6, *args, **kwargs):
        super(Adgd, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.estimate_new_stepsize()
        self.x_old = copy.deepcopy(self.x)
        self.grad_old = copy.deepcopy(self.grad)
        self.x -= self.lr * self.grad
        
    def estimate_new_stepsize(self):
        if self.grad_old is not None:
            L = self.loss.norm(self.grad-self.grad_old) / self.loss.norm(self.x-self.x_old)
            if L == 0:
                lr_new = np.sqrt(1+self.theta) * self.lr
            else:
                lr_new = min(np.sqrt(1+self.theta) * self.lr, 0.5/L)
            self.theta = lr_new / self.lr
            self.lr = lr_new
    
    def init_run(self, *args, **kwargs):
        super(Adgd, self).init_run(*args, **kwargs)
        self.lr = self.lr0
        self.trace.lrs = [self.lr]
        self.theta = 1e12
        self.grad_old = None
        
    def update_trace(self):
        super(Adgd, self).update_trace()
        self.trace.lrs.append(self.lr)
