import scipy
import numpy as np
import warnings

from numba import njit
from sklearn.utils.extmath import row_norms, safe_sparse_dot

from .loss_oracle import Oracle
from .utils import safe_sparse_add, safe_sparse_multiply, safe_sparse_norm, safe_sparse_inner_prod


@njit
def logsig(x):
    """
    Compute the log-sigmoid function component-wise.
    See http://fa.bianp.net/blog/2019/evaluate_logistic/ for more details.
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


class LogisticRegression(Oracle):
    """
    Logistic regression oracle that returns loss values, gradients, Hessians,
    their stochastic analogues as well as smoothness constants. Supports both
    sparse and dense iterates but is far from optimal for dense vectors.
    """
    
    def __init__(self, A, b, store_mat_vec_prod=True, *args, **kwargs):
        super(LogisticRegression, self).__init__(*args, **kwargs)
        self.A = A
        b = np.asarray(b)
        b_unique = np.unique(b)
        # check that only two unique values exist in b
        if len(b_unique) == 1:
            warnings.warn('The labels have only one unique value.')
            self.b = b
        if len(b_unique) > 2:
            raise ValueError('The number of classes must be no more than 2 for binary classification.')
        self.b = b
        if len(b_unique) == 2 and (b_unique != [0, 1]).any():
            if (b_unique == [1, 2]).all():
                print('The passed labels have values in the set {1, 2}. Changing them to {0, 1}')
                self.b = b - 1
            elif (b_unique == [-1, 1]).all():
                print('The passed labels have values in the set {-1, 1}. Changing them to {0, 1}')
                self.b = (b+1) / 2
            else:
                print(f'Changing the labels from {b[0]} to 1s and the rest to 0s')
                self.b = 1. * (b == b[0])
        self.store_mat_vec_prod = store_mat_vec_prod
        
        self.n, self.dim = A.shape
        self.x_last = 0.
        self._mat_vec_prod = np.zeros(self.n)
    
    def _value(self, x):
        Ax = self.mat_vec_product(x)
        regularization = 0
        if self.l2 != 0:
            regularization = self.l2 / 2 * safe_sparse_norm(x)**2
        return np.mean(safe_sparse_multiply(1-self.b, Ax)-logsig(Ax)) + regularization
    
    def partial_value(self, x, idx, include_reg=True, normalization=None, return_idx=False):
        batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        Ax = self.A[idx] @ x
        if scipy.sparse.issparse(Ax):
            Ax = Ax.toarray().ravel()
        regularization = 0
        if include_reg:
            regularization = self.l2 / 2 * safe_sparse_norm(x)**2
        value = np.sum(safe_sparse_multiply(1-self.b[idx], Ax)-logsig(Ax))/normalization + regularization
        if return_idx:
            return (value, idx)
        return value
    
    def gradient(self, x):
        Ax = self.mat_vec_product(x)
        activation = scipy.special.expit(Ax)
        if self.l2 == 0:
            grad = self.A.T@(activation-self.b)/self.n
        else:
            grad = safe_sparse_add(self.A.T@(activation-self.b)/self.n, self.l2*x)
        if scipy.sparse.issparse(x):
            grad = scipy.sparse.csr_matrix(grad).T
        return grad
    
    def stochastic_gradient(self, x, idx=None, batch_size=1, replace=False, normalization=None, 
                            importance_sampling=False, p=None, rng=None, return_idx=False):
        """
        normalization (int, optional): this parameter is needed for Shuffling optimizer
            to remove the bias of the last (incomplete) minibatch
        """
        if batch_size is None or batch_size == self.n:
            return (self.gradient(x), np.arange(n)) if return_idx else self.gradient(x)
        if idx is None:
            if rng is None:
                rng = self.rng
            if p is None and importance_sampling:
                if self._importance_probs is None:
                    self._importance_probs = self.individ_smoothness
                    self._importance_probs /= sum(self._importance_probs)
                p = self._importance_probs
            idx = np.random.choice(self.n, size=batch_size, replace=replace, p=p)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            if p is None:
                normalization = batch_size
            else:
                normalization = batch_size * p[idx] * self.n
        A_idx = self.A[idx]
        Ax = A_idx @ x
        if scipy.sparse.issparse(Ax):
            Ax = Ax.toarray().ravel()
        activation = scipy.special.expit(Ax)
        if scipy.sparse.issparse(x):
            error = scipy.sparse.csr_matrix((activation-self.b[idx]) / normalization)
        else:
            error = (activation-self.b[idx]) / normalization
        if not np.isscalar(error):
            grad = self.l2*x + (error@A_idx).T
        else:
            grad = self.l2*x + error*A_idx.T
        if return_idx:
            return (grad, idx)
        return grad
    
    def hessian(self, x):
        Ax = self.mat_vec_product(x)
        activation = scipy.special.expit(Ax)
        weights = activation * (1-activation)
        A_weighted = safe_sparse_multiply(self.A.T, weights)
        return A_weighted@self.A/self.n + self.l2*np.eye(self.dim)
    
    def stochastic_hessian(self, x, idx=None, batch_size=1, replace=False, normalization=None, 
                           rng=None, return_idx=False):
        if batch_size == self.n:
            return (self.hessian(x), np.arange(n)) if return_idx else self.hessian(x)
        if idx is None:
            if rng is None:
                rng = self.rng
            idx = rng.choice(self.n, size=batch_size, replace=replace)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        A_idx = self.A[idx]
        Ax = A_idx @ x
        if scipy.sparse.issparse(Ax):
            Ax = Ax.toarray().ravel()
        activation = scipy.special.expit(Ax)
        weights = activation * (1-activation)
        A_weighted = safe_sparse_multiply(A_idx.T, weights)
        hess = A_weighted@A_idx/normalization + self.l2*np.eye(self.dim)
        if return_idx:
            return (hess, idx)
        return hess
    
    def mat_vec_product(self, x):
        if self.store_mat_vec_prod and self.is_equal(x, self.x_last):
            return self._mat_vec_prod
        Ax = self.A @ x
        if scipy.sparse.issparse(Ax):
            Ax = Ax.toarray()
        Ax = Ax.ravel()
        if self.store_mat_vec_prod:
            self._mat_vec_prod = Ax
            self.x_last = x.copy()
        return Ax
    
    def hess_vec_prod(self, x, v, grad_dif=False, eps=None):
        if grad_dif:
            grad_x = self.gradient(x)
            grad_x_v = self.gradient(x + eps * v)
            return (grad_x_v - grad_x) / eps
        return safe_sparse_dot(self.hessian(x), v)
        
    @property
    def smoothness(self):
        if self._smoothness is not None:
            return self._smoothness
        if self.dim > 20000 and self.n > 20000:
            warnings.warn("The matrix is too large to estimate the smoothness constant, so Frobenius estimate is used instead.")
            if scipy.sparse.issparse(self.A):
                self._smoothness = 0.25*scipy.sparse.linalg.norm(self.A, ord='fro')**2/self.n + self.l2
            else:
                self._smoothness = 0.25*np.linalg.norm(self.A, ord='fro')**2/self.n + self.l2
        else:
            sing_val_max = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)[0]
            self._smoothness = 0.25*sing_val_max**2/self.n + self.l2
        return self._smoothness
    
    @property
    def max_smoothness(self):
        if self._max_smoothness is not None:
            return self._max_smoothness
        max_squared_sum = row_norms(self.A, squared=True).max()
        self._max_smoothness = 0.25*max_squared_sum + self.l2
        return self._max_smoothness
    
    @property
    def average_smoothness(self):
        if self._ave_smoothness is not None:
            return self._ave_smoothness
        ave_squared_sum = row_norms(self.A, squared=True).mean()
        self._ave_smoothness = 0.25*ave_squared_sum + self.l2
        return self._ave_smoothness
    
    def batch_smoothness(self, batch_size):
        "Smoothness constant of stochastic gradients sampled in minibatches"
        L = self.smoothness
        L_max = self.max_smoothness
        L_batch = self.n / (self.n-1) * (1-1/batch_size) * L + (self.n/batch_size-1) / (self.n-1) * L_max
        return L_batch
    
    @property
    def individ_smoothness(self):
        if self._individ_smoothness is not None:
            return self._individ_smoothness
        self._individ_smoothness = row_norms(self.A)
        return self._individ_smoothness
    
    @property
    def hessian_lipschitz(self):
        if self._hessian_lipschitz is not None:
            return self._hessian_lipschitz
        # Estimate the norm of tensor T = sum_i f_i(x)''' * [a_i, a_i, a_i] as ||T|| <= max||a_i|| * max|f_i'''| * ||A||^2
        a_max = row_norms(self.A, squared=False).max()
        A_norm = (self.smoothness - self.l2) * 4
        self._hessian_lipschitz = A_norm * a_max / (6*np.sqrt(3))
        return self._hessian_lipschitz
    
    @staticmethod
    def norm(x, ord=None):
        return safe_sparse_norm(x, ord=ord)
    
    @staticmethod
    def inner_prod(x, y):
        return safe_sparse_inner_prod(x, y)
    
    @staticmethod
    def outer_prod(x, y):
        return np.outer(x, y)
    
    @staticmethod
    def is_equal(x, y):
        if x is None:
            return y is None
        if y is None:
            return False
        x_sparse = scipy.sparse.issparse(x)
        y_sparse = scipy.sparse.issparse(y)
        if (x_sparse and not y_sparse) or (y_sparse and not x_sparse):
            return False
        if not x_sparse and not y_sparse:
            return np.array_equal(x, y)
        if x.nnz != y.nnz:
            return False
        return (x!=y).nnz == 0 
    
    @staticmethod
    def density(x):
        if hasattr(x, "toarray"):
            dty = float(x.nnz) / (x.shape[0]*x.shape[1])
        else:
            dty = 0 if x is None else float((x!=0).sum()) / x.size
        return dty
