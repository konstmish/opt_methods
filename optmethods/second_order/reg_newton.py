import copy
import numpy as np
import warnings

from optmethods.line_search import RegNewtonLS
from optmethods.optimizer import Optimizer


def empirical_hess_lip(grad, grad_old, hess, x, x_old, loss):
    grad_error = grad - grad_old - hess@(x - x_old)
    r2 = loss.norm(x - x_old)**2
    if r2 > 0:
        return 2 * loss.norm(grad_error) / r2
    return np.finfo(float).eps


class RegNewton(Optimizer):
    """
    Regularized Newton algorithm for second-order minimization.
    By default returns the Regularized Newton method from paper
        "Regularized Newton Method with Global O(1/k^2) Convergence"
        https://arxiv.org/abs/2112.02089
    
    Arguments:
        loss (optmethods.loss.Oracle): loss oracle
        identity_coef (float, optional): initial regularization coefficient (default: None)
        hess_lip (float, optional): estimate for the Hessian Lipschitz constant. 
            If not provided, it is estimated or a small value is used (default: None)
        adaptive (bool, optional): use decreasing regularization based on either empirical Hessian-Lipschitz constant
            or a line-search procedure
        line_search (optmethods.LineSearch, optional): a callable line search. If None, line search is intialized
            automatically as an instance of RegNewtonLS (default: None)
        use_line_search (bool, optional): use line search to estimate the Lipschitz constan of the Hessian.
            If adaptive is True, line search will be non-monotonic and regularization may decrease (default: False)
        backtracking (float, optional): backtracking constant for the line search if line_search is None and
            use_line_search is True (default: 0.5)
    """
    def __init__(self, loss, identity_coef=None, hess_lip=None, adaptive=False, line_search=None,
                 use_line_search=False, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        
        self.H = hess_lip / 2
            
        if use_line_search and line_search is None:
            if adaptive:
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, H0=self.H)
            else:
                # use a more optimistic initial estimate since hess_lip is often too optimistic
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, H0=self.H / 100)
        super(RegNewton, self).__init__(loss=loss, line_search=line_search, *args, **kwargs)
        
        self.identity_coef = identity_coef
        self.adaptive = adaptive
        self.use_line_search = use_line_search
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        if self.adaptive and self.hess is not None and not self.use_line_search:
            self.hess_lip /= 2
            empirical_lip = empirical_hess_lip(self.grad, self.grad_old, self.hess, self.x, self.x_old, self.loss)
            self.hess_lip = max(self.hess_lip, empirical_lip)
        self.hess = self.loss.hessian(self.x)
        
        if self.use_line_search:
            self.x = self.line_search(self.x, self.grad, self.hess)
        else:
            if self.adaptive:
                self.H = self.hess_lip / 2
            grad_norm = self.loss.norm(self.grad)
            self.identity_coef = (self.H * grad_norm)**0.5
            self.x_old = copy.deepcopy(self.x)
            self.grad_old = copy.deepcopy(self.grad)
            delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
            self.x += delta_x
        
    def init_run(self, *args, **kwargs):
        super(RegNewton, self).init_run(*args, **kwargs)
        self.x_old = None
        self.hess = None
        self.trace.lrs = []
        
    def update_trace(self, *args, **kwargs):
        super(RegNewton, self).update_trace(*args, **kwargs)
        if not self.use_line_search:
            self.trace.lrs.append(1 / self.identity_coef)
