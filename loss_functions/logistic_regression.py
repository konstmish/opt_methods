import scipy
import numpy as np

import numpy.linalg as la
from sklearn.utils.extmath import row_norms, safe_sparse_dot

from .loss_oracle import Oracle
from .utils import safe_sparse_add, safe_sparse_multiply, safe_sparse_norm, safe_sparse_inner_prod


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
    Logistic regression oracle that returns loss values, gradients and Hessians.
    """
    
    def __init__(self, A, b, store_mat_vec_prod=True, *args, **kwargs):
        super(LogisticRegression, self).__init__(*args, **kwargs)
        self.A = A
        b = np.asarray(b)
        if (np.unique(b) == [1, 2]).all():
            # Transform labels {1, 2} to {0, 1}
            self.b = b - 1
        elif (np.unique(b) == [-1, 1]).all():
            # Transform labels {-1, 1} to {0, 1}
            self.b = (b+1) / 2
        else:
            assert (np.unique(b) == [0, 1]).all()
            self.b = b
        self.n, self.dim = A.shape
        self.store_mat_vec_prod = store_mat_vec_prod
        self.x_last = 0.
        self.mat_vec_prod = np.zeros(self.n)
    
    def value_(self, x):
        z = self.mat_vec_product(x)
        regularization = 0
        if self.l1 != 0 or self.l2 != 0:
            regularization = self.l1*safe_sparse_norm(x, ord=1) + self.l2/2*safe_sparse_norm(x)**2
        return np.mean(safe_sparse_multiply(1-self.b, z)-logsig(z)) + regularization
    
    def partial_value(self, x, idx, include_reg=True, normalization=None):
        batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        z = self.A[idx] @ x
        if scipy.sparse.issparse(z):
            z = z.toarray().ravel()
        regularization = 0
        if include_reg:
            regularization = self.l1*safe_sparse_norm(x, ord=1) + self.l2/2*safe_sparse_norm(x)**2
        return np.sum(safe_sparse_multiply(1-self.b[idx], z)-logsig(z))/normalization + regularization
    
    def gradient(self, x):
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)
        grad = safe_sparse_add(self.A.T@(activation-self.b)/self.n, self.l2*x)
        if scipy.sparse.issparse(x):
            grad = scipy.sparse.csr_matrix(grad).T
        return grad
    
    def stochastic_gradient(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        """
        normalization is needed for Shuffling optimizer
            to remove the bias of the last (incomplete) minibatch
        """
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        z = self.A[idx] @ x
        if scipy.sparse.issparse(z):
            z = z.toarray().ravel()
        activation = scipy.special.expit(z)
        stoch_grad = safe_sparse_add(self.A[idx].T@(activation-self.b[idx])/normalization, self.l2*x)
        if scipy.sparse.issparse(x):
            stoch_grad = scipy.sparse.csr_matrix(stoch_grad).T
        return stoch_grad
    
    def hessian(self, x):
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)
        weights = activation * (1-activation)
        A_weighted = safe_sparse_multiply(self.A.T, weights)
        return A_weighted@self.A/self.n + self.l2*np.eye(self.dim)
    
    def stochastic_hessian(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        z = self.A[idx] @ x
        if scipy.sparse.issparse(z):
            z = z.toarray().ravel()
        activation = scipy.special.expit(z)
        weights = activation * (1-activation)
        A_weighted = safe_sparse_multiply(self.A[idx].T, weights)
        return A_weighted@self.A[idx]/normalization + self.l2*np.eye(self.dim)
    
    def mat_vec_product(self, x):
        if not self.store_mat_vec_prod or safe_sparse_norm(safe_sparse_add(x, -self.x_last)) != 0:
            z = self.A @ x
            if scipy.sparse.issparse(z):
                z = z.toarray().ravel()
            if self.store_mat_vec_prod:
                self.mat_vec_prod = z
                self.x_last = x.copy()
        
        return self.mat_vec_prod
    
    def hess_vec_prod(self, x, v, grad_dif=False, eps=None):
        if grad_dif:
            grad_x = self.gradient(x)
            grad_x_v = self.gradient(x + eps * v)
            return (grad_x_v - grad_x) / eps
        return safe_sparse_dot(self.hessian(x), v)
        
    def smoothness(self):
        if scipy.sparse.issparse(self.A):
            sing_val_max = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)[0]
            return 0.25*sing_val_max**2/self.n + self.l2
        else:
            covariance = self.A.T@self.A/self.n
            return 0.25*np.max(la.eigvalsh(covariance)) + self.l2
    
    def max_smoothness(self):
        max_squared_sum = row_norms(self.A, squared=True).max()
        return 0.25*max_squared_sum + self.l2
    
    def average_smoothness(self):
        ave_squared_sum = row_norms(self.A, squared=True).mean()
        return 0.25*ave_squared_sum + self.l2
    
    def batch_smoothness(self, batch_size):
        "Smoothness constant of stochastic gradients sampled in minibatches"
        L = self.smoothness()
        L_max = self.max_smoothness()
        L_batch = self.n / (self.n-1) * (1-1/batch_size) * L + (self.n/batch_size-1) / (self.n-1) * L_max
        return L_batch
    
    @staticmethod
    def norm(x):
        return safe_sparse_norm(x)
    
    @staticmethod
    def inner_prod(x, y):
        return safe_sparse_inner_prod(x, y)
    
    @staticmethod
    def outer_prod(x, y):
        return np.outer(x, y)
    
    @staticmethod
    def is_equal(x, y):
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
