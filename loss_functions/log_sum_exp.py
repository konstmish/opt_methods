import scipy
import numpy as np
import warnings

import numpy.linalg as la
from sklearn.utils.extmath import row_norms

from .loss_oracle import Oracle


class LogSumExp(Oracle):
    """
    Logarithm of the sum of exponentials plus two optional quadratic terms:
        log(sum_{i=1}^n exp(<a_i, x>-b_i)) + 1/2*||Ax - b||^2 + l2/2*||x||^2
    See, for instance,
        https://arxiv.org/pdf/2002.00657.pdf
        https://arxiv.org/pdf/2002.09403.pdf
    for examples of using the objective to benchmark second-order methods.
    
    Due to the potential under- and overflow, log-sum-exp and softmax
    functions might be unstable. This implementation has not been tested
    for stability and an alternative choice of functions may lead to
    more precise results. See
        https://academic.oup.com/imajna/advance-article/doi/10.1093/imanum/draa038/5893596
    for a discussion of possible ways to increase stability.
    
    Sparse matrices are currently not supported as it is not clear if it is relevant to practice.
    
    Arguments:
        least_squares_term (bool, optional): add term 0.5*||Ax-b||^2 to the objective (default: False)
    """
    
    def __init__(self, least_squares_term=False, A=None, b=None, n=None, dim=None, store_mat_vec_prod=True,
                 store_AT_A=True, store_softmax=True, *args, **kwargs):
        super(LogSumExp, self).__init__(*args, **kwargs)
        self.least_squares_term = least_squares_term
        self.A = A
        self.b = np.asarray(b)
        if b is None:
            self.b = self.rng.normal(-1, 1, size=n)
        if A is None:
            self.A = self.rng.uniform(-1, 1, size=(n, dim))
            self.store_mat_vec_prod = False
            self.store_softmax = False
            self.A -= self.gradient(np.zeros(dim))
            self.value(np.zeros(dim))
        self.store_mat_vec_prod = store_mat_vec_prod
        self.store_AT_A = store_AT_A
        self.store_softmax = store_softmax
        
        self.n, self.dim = self.A.shape
        self.x_last_mv = 0.
        self.x_last_soft = 0.
        self._mat_vec_prod = np.zeros(self.n)
    
    def _value(self, x):
        Ax = self.mat_vec_product(x)
        regularization = 0
        if self.l2 != 0:
            regularization = self.l2/2 * self.norm(x)**2
        if self.least_squares_term:
            regularization += 1/2 * np.linalg.norm(Ax)**2
        return scipy.special.logsumexp(Ax - self.b) + regularization
    
    def gradient(self, x):
        Ax = self.mat_vec_product(x)
        softmax = self.softmax(x=x, Ax=Ax)
        if self.least_squares_term:
            grad = (softmax + Ax) @ self.A
        else:
            grad = softmax @ self.A
            
        if self.l2 == 0:
            return grad
        return grad + self.l2 * x
    
    def hessian(self, x):
        Ax = self.mat_vec_product(x)
        hess1 = self.A.T * (self.softmax + 1) @ self.A
        g = softmax @ self.A
        hess2 = -np.outer(g, g)
        return hess1 + hess2 + self.l2 * np.eye(self.dim)
    
    def stochastic_hessian(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        pass
    
    def mat_vec_product(self, x):
        if self.store_mat_vec_prod and self.is_equal(x, self.x_last_mv):
            return self._mat_vec_prod
        
        Ax = self.A @ x
        if self.store_mat_vec_prod:
            self._mat_vec_prod = Ax
            self.x_last_mv = x.copy()
        return Ax
    
    def softmax(self, x=None, Ax=None):
        if x is None and Ax is None:
            raise ValueError("Either x or Ax must be provided to compute softmax.")
        if Ax is None:
            Ax = self.mat_vec_product(x)
        if self.store_softmax and self.is_equal(x, self.x_last_soft):
            return self._softmax
        
        softmax = scipy.special.softmax(Ax - self.b)
        if self.store_softmax and x is not None:
            self._softmax = softmax
            self.x_last_soft = x.copy()
        return softmax
    
    def hess_vec_prod(self, x, v, grad_dif=False, eps=None):
        pass
        
    @property
    def smoothness(self):
        if self._smoothness is None:
            self._smoothness = 2*np.linalg.norm(self.A) + self.l2
        return self._smoothness
    
    @property
    def hess_lip(self):
        if self._hess_lip is None:
            self._hess_lip = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)[0]**2
        return self._hess_lip
    
    @staticmethod
    def norm(x):
        return np.linalg.norm(x)
    
    @staticmethod
    def inner_prod(x, y):
        return x @ y
    
    @staticmethod
    def outer_prod(x, y):
        return np.outer(x, y)
    
    @staticmethod
    def is_equal(x, y):
        return np.array_equal(x, y)
