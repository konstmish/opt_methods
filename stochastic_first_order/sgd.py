import numpy as np

from scipy.sparse import csr_matrix

from optimizer import StochasticOptimizer


class Sgd(StochasticOptimizer):
    """
    Stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        lr_decay_coef (float, optional): the coefficient in front of the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value
            is mu/2, where mu is the strong convexity constant
        lr_decay_power (float, optional): the power to exponentiate the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value is 1 (default: 1)
        it_start_decay (int, optional): how many iterations the step-size is kept constant
            By default, will be set to have about 2.5% of iterations with the step-size equal to lr0
        batch_size (int, optional): the number of samples from the function to be used at each iteration
        avoid_cache_miss (bool, optional): whether to make iterations faster by using chunks of the data
            that are adjacent to each other. Implemented by sampling an index and then using the next
            batch_size samples to obtain the gradient. May lead to slower iteration convergence (default: True)
    """
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None,
                 batch_size=1, avoid_cache_miss=False, importance_sampling=False, *args, **kwargs):
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
        self.importance_sampling = importance_sampling
        
    def step(self):
        if self.avoid_cache_miss:
            i = np.random.choice(self.loss.n)
            idx = np.arange(i, i + self.batch_size)
            idx %= self.loss.n
            self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        else:
            self.grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size, importance_sampling=self.importance_sampling)
        denom_const = 1 / self.lr0
        it_decrease = max(0, self.it-self.it_start_decay)
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*it_decrease**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
        if self.use_prox:
            self.x = self.loss.regularizer.prox(self.x, self.lr)
    
    def init_run(self, *args, **kwargs):
        super(Sgd, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
