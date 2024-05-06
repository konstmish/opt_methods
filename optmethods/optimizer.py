import copy
import numpy as np
import numpy.linalg as la
import random
import scipy
import time

from tqdm.notebook import tqdm

from optmethods.opt_trace import Trace, StochasticTrace


SEED = 42
MAX_SEED = 10000000000
MAX_TOTAL_SEEDS = 1000


class Optimizer:
    """
    Base class for optimization algorithms. It implements the optimization step,
    saves the trace of each run, plots the results. Supports multiple random seeds in case of
    stochastic/randomized algorithms.
    
    Arguments:
        loss (required): an instance of class Oracle, which will be used to produce gradients,
              loss values, or whatever else is required by the optimizer
        trace_len (int, optional): the number of checkpoints that will be stored in the
                                  trace. Larger value may slowdown the runtime (default: 200)
        use_prox (bool, optional): whether the optimizer should treat the regularizer
                                   using prox (default: True)
        tolerance (float, optional): stationarity level at which the method should interrupt.
                                     Stationarity is computed using the difference between
                                     two consecutive iterates(default: 0)
        line_search (optional): an instance of class LineSearch, which is used to tune stepsize,
                                or other parameters of the optimizer (default: None)
        save_first_iterations (int, optional): how many of the very first iterations should be
                                               saved as checkpoints in the trace. Useful when
                                               optimizer converges fast at the beginning 
                                               (default: 5)
        label (string, optional): label to be passed to the Trace attribute (default: None)
        n_seeds (int, optional): The number of different random seeds to use to run the method
                                 multiple times. Ignored if the list of seeds in provided instead.
                                 (default: 1)
        seeds (list, optional): random seeds to be used to create random number generator (RNG).
                                If None, a single random seed 42 will be used (default: None)
        tqdm (bool, optional): whether to use tqdm to report progress of the run (default: True)
    """
    def __init__(self, loss, trace_len=200, use_prox=True, tolerance=0, line_search=None,
                 save_first_iterations=5, label=None, n_seeds=1, seeds=None, tqdm=True):
        self.loss = loss
        self.trace_len = trace_len
        self.use_prox = use_prox and (self.loss.regularizer is not None)
        self.tolerance = tolerance
        self.line_search = line_search
        self.save_first_iterations = save_first_iterations
        self.label = label
        self.tqdm = tqdm
        if n_seeds > MAX_TOTAL_SEEDS:
            raise ValueError(f'At most {MAX_TOTAL_SEEDS} random seeds are supported.')
        
        self.initialized = {}
        self.x_old_tol = None
        self.trace = Trace(loss=loss, label=label)
        self.seeds = seeds
        if seeds is None:
            rng = np.random.default_rng(seed=SEED)
            # to make sure we get the same random seeds, we generate a lot of them
            # and take only the first n_seeds
            self.seeds = rng.choice(MAX_SEED, size=MAX_TOTAL_SEEDS, replace=False)[:n_seeds]
        self.n_seeds = len(self.seeds)
        self.initialized = {seed: False for seed in self.seeds}
        self.finished_seeds = []
        self.seed = None
    
    def run(self, x0, t_max=np.inf, it_max=np.inf, ls_it_max=None, tqdm_seeds=True, tqdm_iterations=True):
        if t_max is np.inf and it_max is np.inf:
            it_max = 100
            print(f'{self.label}: The number of iterations is set to {it_max}.')
        self.t_max = t_max
        self.it_max = it_max
        tqdm_seeds = tqdm_seeds and len(self.seeds) > 1
        if tqdm_seeds:
            pbar_seeds = tqdm(total=len(self.seeds))
        for seed in self.seeds:
            if seed in self.finished_seeds:
                continue
            self.rng = np.random.default_rng(seed)
            self.seed = seed
            loss_seed = self.rng.choice(MAX_SEED)
            self.loss.set_seed(loss_seed)
            self.trace.init_seed()
            if ls_it_max is None:
                self.ls_it_max = it_max
            if not self.initialized[seed]:
                self.init_run(x0)
                self.initialized[seed] = True
                if self.line_search is not None:
                    self.line_search.reset()
                
            it_criterion = self.ls_it_max is not np.inf
            if tqdm_iterations:
                tqdm_total = self.ls_it_max if it_criterion else self.t_max
                tqdm_val = 0
                pbar_it = tqdm(total=tqdm_total)
            while not self.check_convergence():
                if self.tolerance > 0:
                    self.x_old_tol = copy.deepcopy(self.x)
                self.step()
                self.save_checkpoint()
                if tqdm_iterations:
                    if it_criterion and self.line_search is not None:
                        tqdm_val_new = self.ls_it
                    elif it_criterion:
                        tqdm_val_new = self.it
                    else:
                        tqdm_val_new = self.t
                    pbar_it.update(tqdm_val_new - tqdm_val)
                    tqdm_val = tqdm_val_new
            self.trace.append_seed_results(seed)
            self.finished_seeds.append(seed)
            if tqdm_seeds:
                pbar_seeds.update(1)
        self.seed = None

        return self.trace
    
    def add_seeds(self, n_extra_seeds=1, extra_seeds=None):
        rng = np.random.default_rng(seed=SEED)
        if extra_seeds is None:
            n_seeds = len(self.seeds) + n_extra_seeds
            # we generate a lot of random seeds so that the behaviour is the same
            # when we add one seed twice as when we add two seeds
            extra_seeds = rng.choice(MAX_SEED, size=MAX_TOTAL_SEEDS, replace=False)[n_seeds-n_extra_seeds:n_seeds]
        self.seeds += extra_seeds
        self.loss_is_computed = False
        if self.trace.its_converted_to_epochs:
            # TODO: create a bool variable for each seed
            pass
        
    def check_convergence(self):
        no_it_left = self.it >= self.it_max
        if self.line_search is not None:
            no_it_left = no_it_left or (self.line_search.it >= self.ls_it_max)
        no_time_left = time.perf_counter() - self.t_start >= self.t_max
        if self.tolerance > 0:
            tolerance_met = self.x_old_tol is not None and self.loss.norm(self.x-self.x_old_tol) < self.tolerance
        else:
            tolerance_met = False
        return no_it_left or no_time_left or tolerance_met
        
    def step(self):
        # should be implemented separately for each optimizer
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
        self.initialized = {key: False for key in self.seeds}
        self.x_old_tol = None
        self.trace = Trace(loss=loss, label=self.label)
