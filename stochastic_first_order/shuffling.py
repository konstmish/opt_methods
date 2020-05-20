import numpy as np

from optimizer import StochasticOptimizer


class Shuffling(StochasticOptimizer):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, steps_per_permutation=None, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, batch_size=1, *args, **kwargs):
        super(Shuffling, self).__init__(*args, **kwargs)
        self.steps_per_permutation = steps_per_permutation if steps_per_permutation else int(self.loss.n/batch_size)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.batch_size = batch_size
        self.sampled_permutations = 0
        
    def step(self):
        if self.it%self.steps_per_permutation == 0:
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0
            self.sampled_permutations += 1
        idx = self.permutation[self.i:min(self.i+self.batch_size, self.loss.n)]
        self.i += self.batch_size
        if self.i >= self.loss.n:
            self.i = 0
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*(self.it+1)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
    
    def init_run(self, *args, **kwargs):
        super(Shuffling, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.max_smoothness()