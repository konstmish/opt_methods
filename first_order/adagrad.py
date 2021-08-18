import copy
import numpy as np

from optimizer import Optimizer


class Adagrad(Optimizer):
    """
    Implement Adagrad from Duchi et. al, 2011
        "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    This implementation only supports deterministic gradients and use dense vectors.
    
    Arguments:
        primal_dual (boolean, optional): if true, uses the dual averaging method of Nesterov, 
            otherwise uses gradient descent update (default: False)
        lr (float, optional): learning rate coefficient, which needs to be tuned to
            get better performance (default: 1.)
        delta (float, optional): another learning rate parameter, slows down performance if
            chosen too large, so a small value is recommended, otherwise requires tuning (default: 0.)
    """
    def __init__(self, primal_dual=False, lr=1., delta=0., *args, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.primal_dual = primal_dual
        self.lr = lr
        self.delta = delta
        
    def estimate_stepsize(self):
        self.s = np.sqrt(self.s**2 + self.grad**2)
        self.inv_lr = self.delta + self.s
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.estimate_stepsize()
        if self.primal_dual:
            self.sum_grad += self.grad
            self.x = self.x0 - self.lr * np.divide(self.sum_grad, self.inv_lr, out=0. * self.x, where=self.inv_lr != 0)
        else:
            self.x -= self.lr * np.divide(self.grad, self.inv_lr, out=0. * self.x, where=self.inv_lr != 0)
        
    def init_run(self, *args, **kwargs):
        super(Adagrad, self).init_run(*args, **kwargs)
        if type(self.x) is not np.ndarray:
            self.x = self.x.toarray().ravel()
        self.x0 = copy.deepcopy(self.x)
        self.s = 0. * self.x
        self.sum_grad = 0. * self.x
