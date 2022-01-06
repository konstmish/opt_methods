import numpy as np
import scipy

from .utils import safe_sparse_norm


class Regularizer():
    """
    A simple oracle class for regularizers that have
    proximal operator and can be evaluated during the training.
    By default, l1+l2 regularization is implemented.
    """
    def __init__(self, l1=0, l2=0, coef=None):
        self.l1 = l1
        self.l2 = l2
        self.coef = coef
    
    def __call__(self, x):
        return self.value(x)
    
    def value(self, x):
        return self.l1*safe_sparse_norm(x, ord=1) + self.l2/2*safe_sparse_norm(x)**2
        
    def prox_l1(self, x, lr=None):
        abs_x = abs(x)
        if scipy.sparse.issparse(x):
            prox_res = abs_x - abs_x.minimum(self.l1 * lr)
            prox_res.eliminate_zeros()
            prox_res = prox_res.multiply(x.sign())
        else:
            prox_res = abs_x - np.minimum(abs_x, self.l1 * lr)
            prox_res *= np.sign(x)
        return prox_res
    
    def prox_l2(self, x, lr=None):
        return x / (1 + lr * self.l2)
        
    def prox(self, x, lr):
        """
        The proximal operator of l1||x||_1 + l2/2 ||x||^2 is equal
        to the combination of the proximal operator of l1||x||_1 and then
        the proximal operator of l2/2 ||x||^2
        """
        prox_l1 = self.prox_l1(x, lr)
        return self.prox_l2(prox_l1, lr)
