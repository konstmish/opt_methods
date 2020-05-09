import numpy as np
import time

import numpy.linalg as la
import matplotlib.pyplot as plt


class Optimizer:
    """
    Base class for optimization algorithms. Provides methods to run them,
    save the trace and plot the results.
    """
    def __init__(self, loss, t_max=np.inf, it_max=np.inf, output_size=500, tolerance=0, verbose=False):
        if t_max is np.inf and it_max is np.inf:
            it_max = 100
            print('The number of iterations is set to 100.')
        self.loss = loss
        self.t_max = t_max
        self.it_max = it_max
        self.output_size = output_size
        self.tolerance = tolerance
        self.verbose = verbose
        self.first_run = True
        self.x_old = None
    
    def run(self, x0):
        if self.first_run:
            self.init_run(x0)
        else:
            self.ts = list(self.ts)
            self.its = list(self.its)
            self.xs = list(self.xs)
        self.first_run = False
        while not self.check_convergence():
            self.x_old = self.x.copy()
            self.step()
            self.save_checkpoint()
            assert np.isfinite(self.x).all()

        self.ts = np.array(self.ts)
        self.its = np.array(self.its)
        self.xs = np.array(self.xs)
        
    def check_convergence(self):
        no_it_left = self.it >= self.it_max
        no_time_left = time.time()-self.t_start >= self.t_max
        tolerance_met = self.x_old is not None and la.norm(self.x-self.x_old) <= self.tolerance
        return no_it_left or no_time_left or tolerance_met
        
    def step(self):
        pass
            
    def init_run(self, x0):
        self.dim = len(x0)
        self.x = x0.copy()
        self.xs = [x0.copy()]
        self.its = [0]
        self.ts = [0]
        self.it = 0
        self.t = 0
        self.t_start = time.time()
        self.time_progress = 0
        self.iterations_progress = 0
        self.max_progress = 0
        self.losses = None
        
    def save_checkpoint(self, first_iterations=10):
        self.it += 1
        self.t = time.time() - self.t_start
        self.time_progress = int((self.output_size-first_iterations) * self.t / self.t_max)
        self.iterations_progress = int((self.output_size-first_iterations) * (self.it / self.it_max))
        if (max(self.time_progress, self.iterations_progress) > self.max_progress) or (self.it <= first_iterations):
            self.update_logs()
        self.max_progress = max(self.time_progress, self.iterations_progress)
        
    def update_logs(self):
        self.xs.append(self.x.copy())
        self.ts.append(self.t)
        self.its.append(self.it)
    
    def compute_loss_of_iterates(self):
        if self.losses is None:
            self.losses = np.array([self.loss.value(x) for x in self.xs])
        elif self.verbose:
            print('Losses have already been computed. Set .losses = None to recompute')
    
    def plot_losses(self, label='', marker=',', f_opt=None, markevery=None):
        if self.losses is None:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = np.min(self.losses)
        if markevery is None:
            markevery = max(1, len(self.losses) // 20)
        plt.plot(self.its, self.losses - f_opt, label=label, marker=marker, markevery=markevery)
        
    def plot_distances(self, label='', marker='', x_opt=None, markevery=None):
        if x_opt is None:
            if self.losses is None:
                x_opt = self.xs[-1]
            else:
                i_min = np.argmin(self.losses)
                x_opt = self.xs[i_min]
        if markevery is None:
            markevery = max(1, len(self.xs) // 20)
        plt.plot(self.its, la.norm(self.xs-x_opt, axis=1), label=label, marker=marker, markevery=markevery)
