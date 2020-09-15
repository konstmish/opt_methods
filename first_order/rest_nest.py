import numpy as np

from optimizer import Optimizer


class RestNest(Optimizer):
    """
    Accelerated gradient descent with constant learning rate.
    For details, see, e.g., http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=None, it_before_first_rest=10, func_condition=False, *args, **kwargs):
        super(RestNest, self).__init__(*args, **kwargs)
        self.lr = lr
        self.it_before_first_rest = it_before_first_rest
        self.func_condition = func_condition
        
    def step(self):
        self.x_nest_old = self.x_nest.copy()
        self.grad = self.loss.gradient(self.x)
        self.x_nest = self.x - self.lr*self.grad
        
        restart = False
        if self.it_without_rest >= self.it_before_first_rest:
            if self.func_condition:
                potential = self.loss.value(self.x_nest)
            else:
                potential = self.loss.norm(self.x_nest - self.x)
            restart = potential > self.potential_old
            self.potential_old = potential
        if restart:
            self.n_restarts += 1
            self.alpha = 1.
            self.it_without_rest = 0
            self.potential_old = np.inf
        else:
            self.it_without_rest += 1
        alpha_new = 0.5 * (1 + np.sqrt(1 + 4*self.alpha**2))
        self.momentum = (self.alpha - 1) / alpha_new
        self.alpha = alpha_new
        self.x = self.x_nest + self.momentum*(self.x_nest-self.x_nest_old)
    
    def init_run(self, *args, **kwargs):
        super(RestNest, self).init_run(*args, **kwargs)
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness()
        self.x_nest = self.x.copy()
        self.alpha = 1.
        self.it_without_rest = 0
        self.potential_old = np.inf
        self.n_restarts = 0
