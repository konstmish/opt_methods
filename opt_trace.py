import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import warnings


class Trace:
    """
    Stores the logs of running an optimization method
    and plots the trajectory.
    """
    def __init__(self, loss):
        self.loss = loss
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = []
        self.its_converted_to_epochs = False
        self.ls_its = None
    
    def compute_loss_of_iterates(self):
        if len(self.loss_vals) == 0:
            self.loss_vals = np.asarray([self.loss.value(x) for x in self.xs])
        else:
            print('Loss values have already been computed. Set .loss_vals = [] to recompute')
    
    def convert_its_to_epochs(self, batch_size=1):
        if self.its_converted_to_epochs:
            warnings.warn('The iteration count has already been converted to epochs.')
            return
        its_per_epoch = self.loss.n / batch_size
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
          
    def plot_losses(self, its=None, f_opt=None, markevery=None, ls_its=False, *args, **kwargs):
        if its is None:
            its = self.ls_its if ls_its and self.ls_its is not None else self.its
        if len(self.loss_vals) == 0:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = self.loss.f_opt
        if markevery is None:
            markevery = max(1, len(self.loss_vals)//20)
        plt.plot(its, self.loss_vals - f_opt, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$f(x)-f^*$')
        
    def plot_distances(self, its=None, x_opt=None, markevery=None, ls_its=False, *args, **kwargs):
        if its is None:
            its = self.ls_its if ls_its and self.ls_its is not None else self.its
        if x_opt is None:
            if self.loss.x_opt is None:
                x_opt = self.xs[-1]
            else:
                x_opt = self.loss.x_opt
        if markevery is None:
            markevery = max(1, len(self.xs)//20)
        dists = [self.loss.norm(x-x_opt)**2 for x in self.xs]
        its = self.ls_its if ls_its and self.ls_its else self.its
        plt.plot(its, dists, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')
        
    @property
    def best_loss_value(self):
        if len(self.loss_vals) == 0:
            self.compute_loss_of_iterates()
        return np.min(self.loss_vals)
        
    def save(self, file_name, path='./results/'):
        # To make the dumped file smaller, remove the loss
        loss_ref_copy = self.loss
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)
        self.loss = loss_ref_copy
        
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
        
        
class StochasticTrace:
    """
    Class that stores the logs of running a stochastic
    optimization method and plots the trajectory.
    """
    def __init__(self, loss):
        self.loss = loss
        self.xs_all = {}
        self.ts_all = {}
        self.its_all = {}
        self.loss_vals_all = {}
        self.its_converted_to_epochs = False
        self.loss_is_computed = False
        
    def init_seed(self):
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = None
        
    def append_seed_results(self, seed):
        self.xs_all[seed] = self.xs.copy()
        self.ts_all[seed] = self.ts.copy()
        self.its_all[seed] = self.its.copy()
        self.loss_vals_all[seed] = self.loss_vals.copy() if self.loss_vals else None
    
    def compute_loss_of_iterates(self):
        for seed, loss_vals in self.loss_vals_all.items():
            if loss_vals is None:
                self.loss_vals_all[seed] = np.asarray([self.loss.value(x) for x in self.xs_all[seed]])
            else:
                print("""Loss values for seed {} have already been computed. 
                      Set .loss_vals_all[{}] = [] to recompute""".format(seed, seed))
        self.loss_is_computed = True
    
    @property
    def best_loss_value(self):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        return np.min([np.min(loss_vals) for loss_vals in self.loss_vals_all.values()])
    
    def convert_its_to_epochs(self, batch_size=1):
        if self.its_converted_to_epochs:
            return
        its_per_epoch = self.loss.n / batch_size
        for seed, its in self.its_all.items():
            self.its_all[seed] = np.asarray(its) / its_per_epoch
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
        
    def plot_losses(self, its=None, f_opt=None, log_std=True, markevery=None, alpha=0.25, *args, **kwargs):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        if its is None:
            its = np.mean([np.asarray(its_) for its_ in self.its_all.values()], axis=0)
        if f_opt is None:
            f_opt = self.loss.f_opt
        if log_std:
            y_log = [np.log(loss_vals-f_opt) for loss_vals in self.loss_vals_all.values()]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            lower, upper = np.exp(y_log_ave - y_log_std), np.exp(y_log_ave + y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = [loss_vals-f_opt for loss_vals in self.loss_vals_all.values()]
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            lower, upper = y_ave - y_std, y_ave + y_std
        if markevery is None:
            markevery = max(1, len(y_ave)//20)
            
        plot = plt.plot(its, y_ave, markevery=markevery, *args, **kwargs)
        if len(self.loss_vals_all.keys()) > 1:
            plt.fill_between(its, lower, upper, alpha=alpha, color=plot[0].get_color())
        plt.ylabel(r'$f(x)-f^*$')
        
    def plot_distances(self, its=None, x_opt=None, log_std=True, markevery=None, alpha=0.25, *args, **kwargs):
        if its is None:
            its = np.mean([np.asarray(its_) for its_ in self.its_all.values()], axis=0)
        if x_opt is None:
            if self.loss.x_opt is None:
                x_opt = self.xs[-1]
            else:
                x_opt = self.loss.x_opt
        
        dists = [np.asarray([self.loss.norm(x-x_opt)**2 for x in xs]) for xs in self.xs_all.values()]
        if log_std:
            y_log = [np.log(dist) for dist in dists]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            lower, upper = np.exp(y_log_ave - y_log_std), np.exp(y_log_ave + y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = dists
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            lower, upper = y_ave - y_std, y_ave + y_std
        if markevery is None:
            markevery = max(1, len(y_ave)//20)
            
        plot = plt.plot(its, y_ave, markevery=markevery, *args, **kwargs)
        if len(self.loss_vals_all.keys()) > 1:
            plt.fill_between(it_ave, lower, upper, alpha=alpha, color=plot[0].get_color())
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')
        
    def save(self, file_name, path='./results/'):
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        f = open(path + file_name, 'wb')
        pickle.dump(self, f)
        f.close()
        
    @classmethod
    def from_pickle(cls, path, loss):
        if not os.path.isfile(path):
            return None
        with open(path, 'rb') as f:
            trace = pickle.load(f)
            trace.loss = loss
        return trace
