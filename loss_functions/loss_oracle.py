import copy
import numpy as np
import warnings

from .regularizer import Regularizer
        

class Oracle():
    """
    Base class for all objectives. Can provide objective values,
    gradients and its Hessians as functions that take parameters as input.
    Takes as input the values of l1 and l2 regularization.
    """
    def __init__(self, l1=0, l2=0, l2_in_prox=False, regularizer=None):
        if l1 < 0.0:
            raise ValueError("Invalid value for l1 regularization: {}".format(l1))
        if l2 < 0.0:
            raise ValueError("Invalid value for l2 regularization: {}".format(l2))
        if l2 == 0. and l2_in_prox:
            warnings.warn("The value of l2 is set to 0, so l2_in_prox is changed to False.")
            l2_in_prox = False
        self.l1 = l1
        self.l2 = 0 if l2_in_prox else l2
        self.l2_in_prox = l2_in_prox
        self.x_opt = None
        self.f_opt = np.inf
        self.regularizer = regularizer
        if (l1 > 0 or l2_in_prox) and regularizer is None:
            l2_prox = l2 if l2_in_prox else 0
            self.regularizer = Regularizer(l1=l1, l2=l2_prox)
        self._smoothness = None
        self._max_smoothness = None
        self._ave_smoothness = None
        self._importance_probs = None
        self._individ_smoothness = None
    
    def value(self, x):
        value = self._value(x)
        if self.regularizer is not None:
            value += self.regularizer(x)
        if value < self.f_opt:
            self.x_opt = copy.deepcopy(x)
            self.f_opt = value
        return value
    
    def gradient(self, x):
        pass
    
    def hessian(self, x):
        pass
    
    def hess_vec_prod(self, x, v, grad_dif=False, eps=None):
        pass
    
    @property
    def smoothness(self):
        pass
    
    @property
    def max_smoothness(self):
        pass
    
    @property
    def average_smoothness(self):
        pass

    def batch_smoothness(self, batch_size):
        pass
    
    @staticmethod
    def norm(x):
        pass
    
    @staticmethod
    def inner_prod(x, y):
        pass
    
    @staticmethod
    def outer_prod(x, y):
        pass
    
    @staticmethod
    def is_equal(x, y):
        pass
