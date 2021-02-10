import copy
import numpy as np

from optimizer import Optimizer


class RestNest(Optimizer):
    """
    Accelerated gradient descent with constant learning rate.
    For details, see, e.g., https://arxiv.org/abs/1204.3982
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        it_before_first_rest (int, optional): number of iterations with increasing momentum when
                                                  restart is not allowed to happen (default: 10)
        func_condition (bool, optional): whether objective (function) decrease should
                                             be checked as restart condition (default: False)
        doubling (bool, optional): instead of checking conditions, just double 
                                       the number of iterations until next restart
                                       after every new restart (default: False)
    """
    def __init__(self, lr=None, it_before_first_rest=10, func_condition=False, doubling=False, *args, **kwargs):
        super(RestNest, self).__init__(*args, **kwargs)
        self.lr = lr
        self.it_before_first_rest = it_before_first_rest
        self.func_condition = func_condition
        self.doubling = doubling
        
    def step(self):
        self.x_old = copy.deepcopy(self.x)
        self.grad = self.loss.gradient(self.x_nest)
        self.x = self.x_nest - self.lr*self.grad
        if self.use_prox:
            self.x = self.loss.regularizer.prox(self.x, self.lr)
        if self.restart_condition():
            self.n_restarts += 1
            self.alpha = 1.
            self.it_without_rest = 0
            self.potential_old = np.inf
        else:
            self.it_without_rest += 1
        alpha_new = 0.5 * (1 + np.sqrt(1 + 4*self.alpha**2))
        self.momentum = (self.alpha - 1) / alpha_new
        self.alpha = alpha_new
        self.x_nest = self.x + self.momentum*(self.x-self.x_old)
        
    def restart_condition(self):
        if self.it_without_rest < self.it_before_first_rest:
            return False
        if self.doubling:
            if self.it_without_rest >= self.it_until_rest:
                self.it_until_rest *= 2
                return True
            return False
        if self.func_condition:
            potential = self.loss.value(self.x)
            restart = potential > self.potential_old
            self.potential_old = potential
            return restart
        direction_is_bad = self.loss.inner_prod(self.x - self.x_old, self.grad) > 0
        return direction_is_bad
    
    def init_run(self, *args, **kwargs):
        super(RestNest, self).init_run(*args, **kwargs)
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness
        self.x_nest = self.x
        self.alpha = 1.
        self.it_without_rest = 0
        self.potential_old = np.inf
        self.n_restarts = 0
        if self.doubling:
            self.it_until_rest = self.it_before_first_rest
