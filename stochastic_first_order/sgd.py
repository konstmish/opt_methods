import numpy as np

from optimizer import StochasticOptimizer


class Sgd(StochasticOptimizer):
    """
    Stochastic gradient descent with constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, batch_size=1, *args, **kwargs):
        super(Sgd, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.batch_size = batch_size
        
    def step(self):
        self.grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
        self.lr = self.lr0 / (1+self.lr_decay_coef*(self.it+1)**self.lr_decay_power)
        self.lr = min(self.lr, self.lr_max)
        self.x -= self.lr * self.grad
    
    def init_run(self, *args, **kwargs):
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.max_smoothness()
        super(Sgd, self).init_run(*args, **kwargs)