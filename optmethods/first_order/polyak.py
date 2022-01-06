import numpy as np

from optmethods.optimizer import Optimizer


class Polyak(Optimizer):
    """
    Polyak adaptive gradient descent, proposed in
        (B. T. Poyal, "Introduction to Optimization")
    which can be accessed, e.g., here:
        https://www.researchgate.net/publication/342978480_Introduction_to_Optimization
    
    Arguments:
        f_opt (float): precise value of the objective's minimum. If an underestimate is given,
            the algorirthm can be unstable; if an overestimate is given, will not converge below
            the overestimate.
        lr_min (float, optional): the smallest step-size, useful when
            an overestimate of the optimal value is given (default: 0)
        lr_max (float, optional): the laregest allowed step-size, useful when
            an underestimate of the optimal value is given (defaul: np.inf)
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
