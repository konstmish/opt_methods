import numpy as np
import numpy.linalg as la
import scipy
import time

from loss_functions import safe_sparse_norm
from opt_trace import Trace, StochasticTrace
from utils import set_seed


class Optimizer:
    """
    Base class for optimization algorithms. Provides methods to run them,
    save the trace and plot the results.
    """
    def __init__(self, loss, t_max=np.inf, it_max=np.inf, trace_len=200, tolerance=0):
        if t_max is np.inf and it_max is np.inf:
            it_max = 100
            print('The number of iterations is set to 100.')
        self.loss = loss
        self.t_max = t_max
        self.it_max = it_max
        self.trace_len = trace_len
        self.tolerance = tolerance
        self.initialized = False
        self.x_old = None
        self.trace = Trace(loss=loss)
    
    def run(self, x0):
        if not self.initialized:
            self.init_run(x0)
            self.initialized = True
        
        while not self.check_convergence():
            if self.tolerance > 0:
                self.x_old = self.x.copy()
            self.step()
            self.save_checkpoint()
            assert scipy.sparse.issparse(self.x) or np.isfinite(self.x).all()

        return self.trace
        
    def check_convergence(self):
        no_it_left = self.it >= self.it_max
        no_time_left = time.time()-self.t_start >= self.t_max
        if self.tolerance > 0:
            tolerance_met = self.x_old is not None and safe_sparse_norm(self.x-self.x_old) < self.tolerance
        else:
            tolerance_met = False
        return no_it_left or no_time_left or tolerance_met
        
    def step(self):
        pass
            
    def init_run(self, x0):
        self.dim = x0.shape[0]
        self.x = x0.copy()
        self.trace.xs = [x0.copy()]
        self.trace.its = [0]
        self.trace.ts = [0]
        self.it = 0
        self.t = 0
        self.t_start = time.time()
        self.time_progress = 0
        self.iterations_progress = 0
        self.max_progress = 0
        
    def save_checkpoint(self, first_iterations=10):
        self.it += 1
        self.t = time.time() - self.t_start
        self.time_progress = int((self.trace_len-first_iterations) * self.t / self.t_max)
        self.iterations_progress = int((self.trace_len-first_iterations) * (self.it / self.it_max))
        if (max(self.time_progress, self.iterations_progress) > self.max_progress) or (self.it <= first_iterations):
            self.update_trace()
        self.max_progress = max(self.time_progress, self.iterations_progress)
        
    def update_trace(self):
        self.trace.xs.append(self.x.copy())
        self.trace.ts.append(self.t)
        self.trace.its.append(self.it)

        
class StochasticOptimizer(Optimizer):
    """
    Base class for stochastic optimization algorithms. 
    The class has the same methods as Optimizer and, in addition, uses
    multiple seeds to run the experiments.
    """
    def __init__(self, loss, n_seeds=1, seeds=None, *args, **kwargs):
        super(StochasticOptimizer, self).__init__(loss=loss, *args, **kwargs)
        self.seeds = seeds
        if not seeds:
            np.random.seed(42)
            self.seeds = [np.random.randint(100000) for _ in range(n_seeds)]
        self.finished_seeds = []
        self.trace = StochasticTrace(loss=loss)
    
    def run(self, *args, **kwargs):
        for seed in self.seeds:
            if seed in self.finished_seeds:
                continue
            set_seed(seed)
            self.trace.init_seed()
            super(StochasticOptimizer, self).run(*args, **kwargs)
            self.trace.append_seed_results(seed)
            self.finished_seeds.append(seed)
            self.initialized = False
        return self.trace
