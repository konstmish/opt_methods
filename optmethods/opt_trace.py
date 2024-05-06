import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import warnings


def plot_with_std(xs, ys, log_std, std_interval_alpha, label='', markevery=None, *args, **kwargs):
    # TODO: add support for ys of different lengths and corresponding to different xs since
    # we might have different line search iterations. Perhaps linear interpolation?
    x = np.mean([np.asarray(x_i) for x_i in xs.values()], axis=0)
    if log_std:
        y_log = [np.log(y) for y in ys]
        y_log_ave = np.mean(y_log, axis=0)
        y_log_std = np.std(y_log, axis=0, ddof=1)
        lower, upper = np.exp(y_log_ave - y_log_std), np.exp(y_log_ave + y_log_std)
        y_ave = np.exp(y_log_ave)
    else:
        y_ave = np.mean(ys, axis=0)
        y_std = np.std(ys, axis=0, ddof=1)
        lower, upper = y_ave - y_std, y_ave + y_std
    if markevery is None:
        markevery = max(1, len(y_ave) // 20)
    plot = plt.plot(x, y_ave, label=label, markevery=markevery, *args, **kwargs)
    plt.fill_between(x, lower, upper, alpha=std_interval_alpha, color=plot[0].get_color())


class Trace:
    """
    Stores the logs of running an optimization method
    and plots the trajectory.
    
    Arguments:
        loss (Oracle): the optimized loss class
        label (string, optional): label for convergence plots (default: None)
    """
    def __init__(self, loss, label=None):
        self.loss = loss
        self.label = label
        
        self.xs_all = {}
        self.ts_all = {}
        self.its_all = {}
        self.loss_vals_all = {}
        self.its_converted_to_epochs = False
        self.loss_is_computed = False
        self.ls_its_all = None
        
    def init_seed(self):
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = None
        self.ls_its = None

    def append_seed_results(self, seed):
        self.xs_all[seed] = self.xs.copy()
        self.ts_all[seed] = self.ts.copy()
        self.its_all[seed] = self.its.copy()
        if self.loss_vals is None:
            self.loss_vals_all[seed] = None
            self.loss_is_computed = False
        else:
            self.loss_vals_all[seed] = self.loss_vals.copy()
    
    def compute_loss_of_iterates(self):
        for seed, loss_vals in self.loss_vals_all.items():
            if loss_vals is None:
                self.loss_vals_all[seed] = np.asarray([self.loss.value(x) for x in self.xs_all[seed]])
            else:
                warnings.warn("""Loss values for seed {} have already been computed. 
                    Set .loss_vals_all[{}] = [] to recompute.""".format(seed, seed))
        self.loss_is_computed = True
    
    def convert_its_to_epochs(self, batch_size=1):
        for seed in self.seeds:
            if self.its_converted_to_epochs[seed]:
                warnings.warn('The iteration count has already been converted to epochs.')
                continue
            its_per_epoch = self.loss.n / batch_size
            self.its = np.asarray(self.its) / its_per_epoch
            self.its_all[seed] = np.asarray(self.its_all[seed]) / its_per_epoch
            self.its_converted_to_epochs[seed] = True
          
    def plot_losses(self, its=None, f_opt=None, log_std=True, std_interval_alpha=0.25, label=None,
                    y_label=None, markevery=None, use_ls_its=True, time=False, *args, **kwargs):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        if its is None:
            if use_ls_its and self.ls_its is not None:
                print(f'Line search iteration counter is used for plotting {label}')
                its_all = self.ls_its_all
            elif time:
                its_all = self.ts_all
            else:
                its_all = self.its_all
        if label is None:
            label = self.label
        if y_label is None:
            y_label = r'$f(x)-f^*$'
        if f_opt is None:
            f_opt = self.loss.f_opt
        n_seeds = len(self.loss_vals_all)
        if n_seeds == 1:
            loss_gaps = list(self.loss_vals_all.values())[0] - f_opt
            its = list(its_all.values())[0]
            if markevery is None:
                markevery = max(1, len(loss_gaps) // 20)
            plt.plot(its, loss_gaps, label=label, markevery=markevery, *args, **kwargs)
        else:
            loss_gaps = [np.asarray(loss_vals) - f_opt for loss_vals in self.loss_vals_all.values()]
            plot_with_std(its_all, loss_gaps, log_std, std_interval_alpha, label, markevery, *args, **kwargs)
        
        plt.ylabel(y_label)
        
    def plot_distances(self, its=None, x_opt=None, log_std=True, std_interval_alpha=0.25, label=None, 
                       y_label=None, markevery=None, use_ls_its=True, time=False, *args, **kwargs):
        if its is None:
            if use_ls_its and self.ls_its is not None:
                print(f'Line search iteration counter is used for plotting {label}')
                its_all = self.ls_its_all
            elif time:
                its_all = self.ts_all
            else:
                its_all = self.its_all
        if x_opt is None:
            if self.loss.x_opt is None:
                x_opt = np.mean([xs[-1] for xs in self.xs_all.values()], axis=0)
            else:
                x_opt = self.loss.x_opt
        if label is None:
            label = self.label
        if y_label is None:
            y_label = r'$\Vert x-x^*\Vert^2$'
        n_seeds = len(self.xs_all)
        if n_seeds == 1:
            dists = [self.loss.norm(x - x_opt) ** 2 for x in self.xs]
            if markevery is None:
                markevery = max(1, len(dists) // 20)
            plt.plot(its, dists, label=label, markevery=markevery, *args, **kwargs)
        else:
            dists = [np.asarray([self.loss.norm(x - x_opt)** 2 for x in xs]) for xs in self.xs_all.values()]
            plot_with_std(its, dists, log_std, std_interval_alpha, label, markevery, *args, **kwargs)
        plt.ylabel(y_label)
        
    @property
    def best_loss_value(self):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        return np.min([np.min(loss_vals) for loss_vals in self.loss_vals_all.values()])
        
    def save(self, file_name=None, path='./results/'):
        if file_name is None:
            file_name = self.label
        if path[-1] != '/':
            path += '/'
        # To make the dumped file smaller, copy the reference to a variable, and remove the loss
        loss = self.loss
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)
        self.loss = loss
        
    @classmethod
    def from_pickle(cls, path, loss=None):
        if not os.path.isfile(path):
            return None
        with open(path, 'rb') as f:
            trace = pickle.load(f)
            trace.loss = loss
        if loss is not None:
            loss.f_opt = min(self.best_loss_value, loss.f_opt)
        return trace
