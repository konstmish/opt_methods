import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from loss_functions import safe_sparse_norm


class Trace:
    """
    Class that stores the logs of running an optimization method
    and plots the trajectory.
    """
    def __init__(self, loss):
        self.loss = loss
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = None
        self.its_converted_to_epochs = False
        self.loss_is_computed = False
    
    def compute_loss_of_iterates(self):
        if self.loss_vals is None:
            self.loss_vals = np.asarray([self.loss.value(x) for x in self.xs])
        else:
            print('Loss values have already been computed. Set .loss_vals = None to recompute')
    
    def convert_its_to_epochs(self, batch_size=1):
        its_per_epoch = self.loss.n / batch_size
        if self.its_converted_to_epochs:
            return
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
          
    def plot_losses(self, f_opt=None, markevery=None, *args, **kwargs):
        if self.loss_vals is None:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = np.min(self.loss_vals)
        if markevery is None:
            markevery = max(1, len(self.loss_vals)//20)
        plt.plot(self.its, self.loss_vals - f_opt, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$f(x)-f^*$')
        
    def plot_distances(self, x_opt=None, markevery=None, *args, **kwargs):
        if x_opt is None:
            if self.loss_vals is None:
                x_opt = self.xs[-1]
            else:
                i_min = np.argmin(self.loss_vals)
                x_opt = self.xs[i_min]
        if markevery is None:
            markevery = max(1, len(self.xs)//20)
        dists = [safe_sparse_norm(x-x_opt)**2 for x in self.xs]
        plt.plot(self.its, dists, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')
        
    def best_loss_value(self):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        return np.min(self.loss_vals)
        
    def save(self, file_name, path='./results/'):
        # To make the dumped file smaller, remove the loss
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        f = open(path + file_name, 'wb')
        pickle.dump(self, f)
        f.close()
        
        
class StochasticTrace:
    """
    Class that stores the logs of running an optimization method
    and plots the trajectory.
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
                      Set .loss_vals_all[{}] = None to recompute""".format(seed, seed))
        self.loss_is_computed = True
    
    def best_loss_value(self):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        return np.min([np.min(loss_vals) for loss_vals in self.loss_vals_all.values()])
    
    def convert_its_to_epochs(self, batch_size=1):
        its_per_epoch = self.loss.n / batch_size
        if self.its_converted_to_epochs:
            return
        for seed, its in self.its_all.items():
            self.its_all[seed] = np.asarray(its) / its_per_epoch
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
        
    def plot_losses(self, f_opt=None, log_std=True, markevery=None, alpha=0.25, *args, **kwargs):
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = self.best_loss_value()
        it_ave = np.mean([np.asarray(its) for its in self.its_all.values()], axis=0)
        if log_std:
            y_log = [np.log(loss_vals-f_opt) for loss_vals in self.loss_vals_all.values()]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            upper, lower = np.exp(y_log_ave + y_log_std), np.exp(y_log_ave - y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = [loss_vals-f_opt for loss_vals in self.loss_vals_all.values()]
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            upper, lower = y_ave + y_std, y_ave - y_std
        if markevery is None:
            markevery = max(1, len(y_ave)//20)
            
        plot = plt.plot(it_ave, y_ave, markevery=markevery, *args, **kwargs)
        if len(self.loss_vals_all.keys()) > 1:
            plt.fill_between(it_ave, upper, lower, alpha=alpha, color=plot[0].get_color())
        plt.ylabel(r'$f(x)-f^*$')
        
    def plot_distances(self, x_opt=None, log_std=True, markevery=None, alpha=0.25, *args, **kwargs):
        if x_opt is None:
            if self.loss_is_computed:
                f_opt = np.inf
                for seed, loss_vals in self.loss_vals_all.items():
                    i_min = np.argmin(loss_vals)
                    if loss_vals[i_min] < f_opt:
                        f_opt = loss_vals[i_min]
                        x_opt = self.xs_all[seed][i_min]
                else:
                    x_opt = self.xs[-1]
        
        it_ave = np.mean([np.asarray(its) for its in self.its_all.values()], axis=0)
        dists = [np.asarray([safe_sparse_norm(x-x_opt)**2 for x in xs]) for xs in self.xs_all.values()]
        if log_std:
            y_log = [np.log(dist) for dist in dists]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            upper, lower = np.exp(y_log_ave + y_log_std), np.exp(y_log_ave - y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = dists
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            upper, lower = y_ave + y_std, y_ave - y_std
        if markevery is None:
            markevery = max(1, len(y_ave)//20)
            
        plot = plt.plot(it_ave, y_ave, markevery=markevery, *args, **kwargs)
        if len(self.loss_vals_all.keys()) > 1:
            plt.fill_between(it_ave, upper, lower, alpha=alpha, color=plot[0].get_color())
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')
        
    def save(self, file_name, path='./results/'):
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        f = open(path + file_name, 'wb')
        pickle.dump(self, f)
        f.close()
