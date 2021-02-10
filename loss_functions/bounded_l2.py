import scipy

from .regularizer import Regularizer


class Boundedl2(Regularizer):
    """
    The bounded l2 regularization is equal to
        R(x) = sum_{i=1}^d x_i^2 / (x_i^2 + 1)
    where x=(x_1, ..., x_d) is from R^d. This penalty is attractive for benchmarking
    purposes since it is smooth (has Lipschitz gradient) and nonconvex.
    
    See 
        https://arxiv.org/pdf/1905.05920.pdf
        https://arxiv.org/pdf/1810.10690.pdf
    for examples of using this penalty for benchmarking.
    """
    def __init__(self, coef):
        self.coef = coef
    
    def value(x, x2=None):
        if not scipy.sparse.issparse(x):
            if x2 is None:
                x2 = x * x
            return self.coef * 0.5 * np.sum(x2 / (x2 + 1))
        if x2 is None:
            x2 = x.multiply(x)
        ones_where_nonzero = x2.sign()
        return self.coef * 0.5 * (x2 / (x2 + ones_where_nonzero)).sum()
    
    def prox(x, lr=None):
        raise NotImplementedError('Exact proximal operator for bounded l2 does not exist. Consider using gradients.')
            
    def grad(self, x):
        if not scipy.sparse.issparse(x):
            return self.coef * x / (x**2+1)**2
        ones_where_nonzero = abs(x.sign())
        x2_plus_one = x.multiply(x) + ones_where_nonzero
        denominator = x2_plus_one.multiply(x2_plus_one)
        return self.coef * x.multiply(ones_where_nonzero / denominator)
        
    @property
    def smoothness(self):
        return self.coef
