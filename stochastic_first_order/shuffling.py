import math
import numpy as np

from optimizer import StochasticOptimizer


class Shuffling(StochasticOptimizer):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, steps_per_permutation=None, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, it_start_decay=None, batch_size=1, *args, **kwargs):
        super(Shuffling, self).__init__(*args, **kwargs)
        self.steps_per_permutation = steps_per_permutation if steps_per_permutation else math.ceil(self.loss.n/batch_size)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size
        self.sampled_permutations = 0
        
    def step(self):
        if self.it%self.steps_per_permutation == 0:
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0
            self.sampled_permutations += 1
        if self.steps_per_permutation is np.inf:
            idx_perm = np.arange(self.i, self.i + self.batch_size)
            idx_perm %= self.loss.n
            normalization = self.batch_size
        else:
            idx_perm = np.arange(self.i, min(self.loss.n, self.i+self.batch_size))
            normalization = self.loss.n / self.steps_per_permutation #works only for RR
        idx = self.permutation[idx_perm]
        self.i += self.batch_size
        self.i %= self.loss.n
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
    
    def init_run(self, *args, **kwargs):
        super(Shuffling, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
