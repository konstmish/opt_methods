import numpy as np

from optimizer import Optimizer


class Polyak(Optimizer):
    """
    Fatkhulin-Polyak adaptive gradient descent 
    https://arxiv.org/pdf/2004.09875.pdf
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, f_opt, lr_min=0, lr_max=np.inf, *args, **kwargs):
        super(Polyak, self).__init__(*args, **kwargs)
        self.f_opt = f_opt
        self.lr_min = lr_min
        self.lr_max = lr_max
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.estimate_new_stepsize()
        self.x -= self.lr * self.grad
        
    def estimate_new_stepsize(self):
        loss_gap = self.loss.value(self.x) - self.f_opt
        self.lr = loss_gap / self.loss.norm(self.grad)**2
        self.lr = min(self.lr, self.lr_max)
        self.lr = max(self.lr, self.lr_min)
    
    def init_run(self, *args, **kwargs):
        super(Polyak, self).init_run(*args, **kwargs)
        self.trace.lrs = []
        
    def update_trace(self):
        super(Polyak, self).update_trace()
        self.trace.lrs.append(self.lr)
