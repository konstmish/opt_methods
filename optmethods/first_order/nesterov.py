import copy
import numpy as np

from optmethods.optimizer import Optimizer


class Nesterov(Optimizer):
    """
    Accelerated gradient descent with constant learning rate.
    For details, see, e.g., 
        http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        strongly_convex (bool, optional): use the variant for strongly convex functions,
            which requires mu to be provided (default: False)
        max_momentum (float, optional): the target value of momentum. If start_with_small_momentum
            is True, the value of momentum in the method will be increased from 0 to the value
            provided in this parameter. Otherwise, it is used in all iterations
        mu (float, optional): strong-convexity constant or a lower bound on it. Ignored if
            momentum is provided (default: 0)
        start_with_small_momentum (bool, optional): momentum gradually increases. Only used if
            strongly_convex is set to True (default: True)
    """
    def __init__(self, lr=None, strongly_convex=False, max_momentum=None, mu=0, 
                 start_with_small_momentum=True, *args, **kwargs):
        super(Nesterov, self).__init__(*args, **kwargs)
        self.lr = lr
        self.max_momentum = max_momentum
        if strongly_convex:
            self.mu = mu
            if mu <= 0 and max_momentum is None:
                raise ValueError("""Mu must be larger than 0 for strongly_convex=True,
                                 invalid value: {}""".format(mu))
        self.strongly_convex = strongly_convex
        self.start_with_small_momentum = start_with_small_momentum
        
    def step(self):
        if not self.strongly_convex or self.start_with_small_momentum:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4*self.alpha**2))
            self.momentum = (self.alpha - 1) / alpha_new
            self.alpha = alpha_new
            self.momentum = min(self.momentum, self.max_momentum)
        else:
            self.momentum = self.max_momentum
        self.x_old = copy.deepcopy(self.x)
        self.grad = self.loss.gradient(self.x_nest)
        self.x = self.x_nest - self.lr*self.grad
        if self.use_prox:
            self.x = self.loss.regularizer.prox(self.x, self.lr)
        self.x_nest = self.x + self.momentum*(self.x-self.x_old)
    
    def init_run(self, *args, **kwargs):
        super(Nesterov, self).init_run(*args, **kwargs)
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness
        self.x_nest = copy.deepcopy(self.x)
        self.alpha = 1.
        if self.strongly_convex and self.max_momentum is None:
            kappa = (1/self.lr)/self.mu
            self.max_momentum = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)
        elif self.max_momentum is None:
            self.max_momentum = 1. - 1e-8
