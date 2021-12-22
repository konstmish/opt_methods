import copy
import numpy as np
import numpy.linalg as la
import scipy
import time

from opt_trace import Trace, StochasticTrace
from utils import set_seed


SEED = 42
MAX_SEED = 10000000


class Optimizer:
    """
    Base class for optimization algorithms. Provides methods to run them,
    save the trace and plot the results.
    
    Arguments:
        label (string, optional): label to be passed to the Trace attribute (default: None)
    """
    def __init__(self, loss, trace_len=200, use_prox=True, tolerance=0, line_search=None,
                 save_first_iterations=5, label=None, seeds=None):
        self.loss = loss
        self.trace_len = trace_len
        self.use_prox = use_prox and (self.loss.regularizer is not None)
        self.tolerance = tolerance
        self.line_search = line_search
        self.save_first_iterations = save_first_iterations
        self.label = label
        
        self.initialized = False
        self.x_old_tol = None
        self.trace = Trace(loss=loss, label=label)
        if seeds is None:
            self.rng = np.random.default_rng(42)
            self.seeds = [42]
        else:
            self.seeds = seeds
        self.finished_seeds = []
    
    def run(self, x0, t_max=np.inf, it_max=np.inf, ls_it_max=None):
        if t_max is np.inf and it_max is np.inf:
            it_max = 100
            print(f'The number of iterations is set to {it_max}.')
        self.t_max = t_max
        self.it_max = it_max
        
        for seed in self.seeds:
            if seed in self.finished_seeds:
                continue
            self.rng = np.random.default_rng(seed)
            if ls_it_max is None:
                self.ls_it_max = it_max
            if not self.initialized:
                self.init_run(x0)
                self.initialized = True

            while not self.check_convergence():
                if self.tolerance > 0:
                    self.x_old_tol = copy.deepcopy(self.x)
                self.step()
                self.save_checkpoint()
            self.finished_seeds.append(seed)
            self.initialized = False

        return self.trace
        
    def check_convergence(self):
        no_it_left = self.it >= self.it_max
        if self.line_search is not None:
            no_it_left = no_it_left or (self.line_search.it >= self.ls_it_max)
        no_time_left = time.perf_counter()-self.t_start >= self.t_max
        if self.tolerance > 0:
            tolerance_met = self.x_old_tol is not None and self.loss.norm(self.x-self.x_old_tol) < self.tolerance
        else:
            tolerance_met = False
        return no_it_left or no_time_left or tolerance_met
        
    def step(self):
        pass
            
    def init_run(self, x0):
        self.dim = x0.shape[0]
        self.x = copy.deepcopy(x0)
        self.trace.xs = [copy.deepcopy(x0)]
        self.trace.its = [0]
        self.trace.ts = [0]
        if self.line_search is not None:
            self.trace.ls_its = [0]
            self.trace.lrs = [self.line_search.lr]
        self.it = 0
        self.t = 0
        self.t_start = time.perf_counter()
        self.time_progress = 0
        self.iterations_progress = 0
        self.max_progress = 0
        if self.line_search is not None:
            self.line_search.reset(self)
        
    def should_update_trace(self):
        if self.it <= self.save_first_iterations:
            return True
        self.time_progress = int((self.trace_len-self.save_first_iterations) * self.t / self.t_max)
        self.iterations_progress = int((self.trace_len-self.save_first_iterations) * (self.it / self.it_max))
        if self.line_search is not None:
            ls_it = self.line_search.it
            self.iterations_progress = max(self.iterations_progress, int((self.trace_len-self.save_first_iterations) * (ls_it / self.it_max)))
        enough_progress = max(self.time_progress, self.iterations_progress) > self.max_progress
        return enough_progress
        
    def save_checkpoint(self):
        self.it += 1
        if self.line_search is not None:
            self.ls_it = self.line_search.it
        self.t = time.perf_counter() - self.t_start
        if self.should_update_trace():
            self.update_trace()
        self.max_progress = max(self.time_progress, self.iterations_progress)
        
    def update_trace(self):
        self.trace.xs.append(copy.deepcopy(self.x))
        self.trace.ts.append(self.t)
        self.trace.its.append(self.it)
        if self.line_search is not None:
            self.trace.ls_its.append(self.line_search.it)
            self.trace.lrs.append(self.line_search.lr)
            
    def compute_loss_of_iterates(self):
        self.trace.compute_loss_of_iterates()
        
    def reset(self):
        self.initialized = False
        self.x_old_tol = None
        self.trace = Trace(loss=loss, label=self.label)

        
class StochasticOptimizer(Optimizer):
    """
    Base class for stochastic optimization algorithms. 
    The class has the same methods as Optimizer and, in addition, uses
    multiple seeds to run the experiments.
    """
    def __init__(self, loss, n_seeds=1, seeds=None, label=None, *args, **kwargs):
        super(StochasticOptimizer, self).__init__(loss=loss, *args, **kwargs)
        self.seeds = seeds
        if not seeds:
            np.random.seed(SEED)
            self.seeds = np.random.choice(MAX_SEED, size=n_seeds, replace=False)
        self.label = label
        self.finished_seeds = []
        self.trace = StochasticTrace(loss=loss, label=label)
        self.seed = None
    
    def run(self, *args, **kwargs):
        for seed in self.seeds:
            if seed in self.finished_seeds:
                continue
            if self.line_search is not None:
                self.line_search.reset()
            set_seed(seed)
            self.seed = seed
            self.trace.init_seed()
            super(StochasticOptimizer, self).run(*args, **kwargs)
            self.trace.append_seed_results(seed)
            self.finished_seeds.append(seed)
            self.initialized = False
        self.seed = None
        return self.trace
    
    def add_seeds(self, n_extra_seeds=1):
        np.random.seed(SEED)
        n_seeds = len(self.seeds) + n_extra_seeds
        self.seeds = np.random.choice(MAX_SEED, size=n_seeds, replace=False)
        self.loss_is_computed = False
        if self.trace.its_converted_to_epochs:
            # TODO: create a bool variable for each seed
            pass
