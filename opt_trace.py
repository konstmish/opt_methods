import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


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
    
    def compute_loss_of_iterates(self):
        if self.loss_vals is None:
            self.loss_vals = np.asarray([self.loss.value(x) for x in self.xs])
        else:
            print('Loss values have already been computed. Set .loss_vals = None to recompute')
          
    def plot_losses(self, f_opt=None, markevery=None, *args, **kwargs):
        if self.loss_vals is None:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = np.min(self.loss_vals)
        if markevery is None:
            markevery = max(1, len(self.loss_vals)//20)
        plt.plot(self.its, self.loss_vals - f_opt, label=label, marker=marker, markevery=markevery)
        plt.ylabel(r'$f(x)-f^*')
        
    def plot_distances(self, x_opt=None, markevery=None, *args, **kwargs):
        if x_opt is None:
            if self.loss_vals is None:
                x_opt = self.xs[-1]
            else:
                i_min = np.argmin(self.loss_vals)
                x_opt = self.xs[i_min]
        if markevery is None:
            markevery = max(1, len(self.xs)//20)
        plt.plot(self.its, la.norm(self.xs-x_opt, axis=1), markevery=markevery, *args, **kwargs)
        plt.ylabel(r'\lvert x-x^*\rvert')

        
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
        
    def init_seed(self):
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = None
        
    def append_seed_results(self, seed):
        self.xs_all[seed] = self.xs.copy()
        self.ts_all[seed] = self.ts.copy()
        self.its_all[seed] = self.its.copy()
        self.loss_vals_all[seed] = self.loss_vals.copy()
    
    def compute_loss_of_iterates(self):
        for seed, loss_vals in self.loss_vals_all.items():
            if loss_vals is None:
                self.loss_vals_all[seed] = np.asarray([self.loss.value(x) for x in self.xs_all[seed]])
            else:
                print("""Loss values for seed {} have already been computed. 
                      Set .loss_vals_all[{}] = None to recompute""".format(seed, seed))
          
    def plot_losses(self, f_opt=None, log_std=True, markevery=None, alpha=0.3, *args, **kwargs):
        for loss_vals in self.loss_vals_all.values():
            if loss_vals is None:
                self.compute_loss_of_iterates()
                break
        if f_opt is None:
            f_opt = np.min([np.min(loss_vals) for self.loss_vals_all.values()])
        if markevery is None:
            markevery = max(1, len(self.loss_vals)//20)
            
        if log_std:
            y = [np.log(loss_vals-f_opt) for loss_vals in self.loss_vals_all.values()]
        else:
            y = [loss_vals-f_opt for loss_vals in self.loss_vals_all.values()]
        y_ave = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
            
        plt.plot(self.its, y_ave, markevery=markevery, *args, **kwargs)
        plt.fill_between(self.its, y_ave + y_std, y_ave - y_std, alpha=alpha, *args, **kwargs)
        plt.ylabel(r'$f(x)-f^*')
        
    def plot_distances(self, x_opt=None, log_std=True, markevery=None, alpha=0.3, *args, **kwargs):
        if x_opt is None:
            if self.loss_vals_all.values()[0] is None:
                x_opt = self.xs[-1]
            else:
                f_opt = np.inf
                for seed, loss_vals in self.loss_vals_all.items():
                    i_min = np.argmin(loss_vals)
                    if loss_vals[i_min] < f_opt:
                        f_opt = loss_vals[i_min]
                        x_opt = self.xs_all[seed][i_min]
        
        if markevery is None:
            markevery = max(1, len(self.xs)//20)
        if log_std:
            y = [np.log(la.norm(xs-x_opt, axis=1)) for xs in self.loss_xs_all.values()]
            y_ave = np.exp(np.mean(y, axis=0))
            y_std = np.exp(np.std(y, axis=0))
        else:
            y = [la.norm(xs-x_opt, axis=1) for xs in self.loss_xs_all.values()]
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            
        plt.plot(self.its, y_ave, markevery=markevery, *args, **kwargs)
        plt.fill_between(self.its, y_ave + y_std, y_ave - y_std, alpha=alpha, *args, **kwargs)
        plt.ylabel(r'\lvert x-x^*\rvert')
