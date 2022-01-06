import numpy as np

from optmethods.optimizer import StochasticOptimizer


class Svrg(StochasticOptimizer):
    """
    Stochastic variance-reduced gradient descent with constant stepsize.
    Reference:
    https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=None, batch_size=1, avoid_cache_miss=False, loopless=True,
                 loop_len=None, restart_prob=None, *args, **kwargs):
        super(Svrg, self).__init__(*args, **kwargs)
        self.lr = lr
        self.batch_size = batch_size
        self.avoid_cache_miss = avoid_cache_miss
        self.loopless = loopless
        self.loop_len = loop_len
        self.restart_prob = restart_prob
        if loopless and restart_prob is None:
            self.restart_prob = batch_size / self.loss.n
        elif not loopless and loop_len is None:
            self.loop_len = self.loss.n // batch_size
        
    def step(self):
        new_loop = self.loopless and np.random.uniform() < self.restart_prob
        if not self.loopless and self.loop_it == self.loop_len:
            new_loop = True
        if new_loop or self.it == 0:
            self.x_old = self.x.copy()
            self.full_grad_old = self.loss.gradient(self.x_old)
            self.vr_grad = self.full_grad_old.copy()
            if not self.loopless:
                self.loop_it = 0
            self.loops += 1
        else:
            if self.avoid_cache_miss:
                i = np.random.choice(self.loss.n)
                idx = np.arange(i, i + self.batch_size)
                idx %= self.loss.n
            else:
                idx = np.random.choice(self.loss.n, size=self.batch_size)
            stoch_grad = self.loss.stochastic_gradient(self.x, idx=idx)
            stoch_grad_old = self.loss.stochastic_gradient(self.x_old, idx=idx)
            self.vr_grad = stoch_grad - stoch_grad_old + self.full_grad_old
            
        self.x -= self.lr * self.vr_grad
        if self.use_prox:
            self.x = self.loss.regularizer.prox(self.x, self.lr)
        self.loop_it += 1
    
    def init_run(self, *args, **kwargs):
        super(Svrg, self).init_run(*args, **kwargs)
        self.loop_it = 0
        self.loops = 0
        if self.lr is None:
            self.lr = 0.5 / self.loss.batch_smoothness(self.batch_size)
