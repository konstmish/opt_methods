import numpy as np

from optimizer import StochasticOptimizer


class Sgd(StochasticOptimizer):
    """
    Stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None,
                 batch_size=1, avoid_cache_miss=True, *args, **kwargs):
        super(Sgd, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size
        self.avoid_cache_miss = avoid_cache_miss
        
    def step(self):
        if self.avoid_cache_miss:
            i = np.random.choice(self.loss.n)
            idx = np.arange(i, i + self.batch_size)
            idx %= self.loss.n
            self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        else:
            self.grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
    
    def init_run(self, *args, **kwargs):
        super(Sgd, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
