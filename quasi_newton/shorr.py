import copy
import numpy as np

from line_search import BestGrid
from optimizer import Optimizer


class Shorr(Optimizer):
    """
    Shor's r-algorithm. For a convergence analysis, see
        https://www.researchgate.net/publication/243084304_The_Speed_of_Shor's_R-algorithm
    In general, the method won't work without a line search. The best line search is 
        probably a grid search that starts at lr=1.
    
    Arguments:
        gamma (float, optional): the closer it is to 1, the faster the Hessian 
                                 estimate changes. Must be from (0, 1) (default: 0.5)
        L (float, optional): an upper bound on the smoothness constant
            to initialize the Hessian estimate
    """
    
    def __init__(self, gamma=0.5, L=None, *args, **kwargs):
        super(Shorr, self).__init__(*args, **kwargs)
        if not 0.0 < gamma < 1.0:
            raise ValueError("Invalid gamma: {}".format(gamma))
        if L is None:
            L = self.loss.smoothness
            if L is None:
                L = 1
        self.gamma = gamma
        self.L = L
        self.B = 1/np.sqrt(self.L) * np.eye(self.loss.dim)
        if self.line_search is None:
            self.line_search = BestGrid(lr0=1.0, start_with_prev_lr=False,
                                        increase_many_times=True)
            self.line_search.loss = self.loss
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        if self.line_search.lr != 1.0:
            # avoid machine precision issues
            self.B *= np.sqrt(self.line_search.lr)
            self.line_search.lr = 1.0
        r = self.B.T @ (self.grad - self.grad_old)
        if self.loss.norm(r) > 0:
            r /= self.loss.norm(r)
        self.B -= self.gamma * self.B @ self.loss.outer_prod(r, r)
        x_new = self.x - self.B @ (self.B.T @ self.grad)
        self.x = self.line_search(self.x, x_new=x_new)
        self.grad_old = copy.deepcopy(self.grad)
    
    def init_run(self, *args, **kwargs):
        super(Shorr, self).init_run(*args, **kwargs)
        self.grad_old = self.loss.gradient(self.x)
        self.x -= 1 / self.L * self.grad_old
        self.save_checkpoint()
