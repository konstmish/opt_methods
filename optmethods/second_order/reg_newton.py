import copy
import numpy as np
import warnings

import numpy.linalg as la

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
        reg_coef (float, optional): regularization coefficient, not used when reg_rule is 'grad' with grad_norm_power=0.5 (default: None)
        hess_lip (float, optional): estimate for the Hessian Lipschitz constant. 
            If not provided, it is estimated or a small value is used (default: None)
        grad_norm_power (float, optional): power of the gradient norm for the coefficient in 
            front of identity matrix. Ignored if adpative is not 'grad'
        adaptive (bool, optional): use decreasing regularization based on either empirical Hessian-Lipschitz constant
            or a line-search procedure
        use_line_search (bool, optional): use line search to estimate the Lipschitz constan of the Hessian.
            If adaptive is True, line search will be non-monotonic and regularization may decrease (default: False)
        line_search (optmethods.LineSearch, optional): a callable line search, here it should be None or
            an instance of RegNewtonLS class.  If None, line search is intialized automatically (default: None)
        backtracking (float, optional): backtracking constant for the line search if line_search is None and
            use_line_search is True. (default: 0.5)
    """
    def __init__(self, loss, reg_coef=None, hess_lip=None, grad_norm_power=0.5, adaptive=False, 
                 line_search=None, use_line_search=False, 
                 backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        if use_line_search and line_search is None:
            if adaptive:
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, lr0=2/self.hess_lip)
            else:
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, lr0=1e3/self.hess_lip)
        super(RegNewton, self).__init__(loss=loss, line_search=line_search, *args, **kwargs)
        self.reg_coef = reg_coef
        self.hess_lip = hess_lip
        self.adaptive = adaptive
        self.use_line_search = use_line_search
        self.grad_norm_power = grad_norm_power
        
        if reg_coef is None:
            self.reg_coef = self.hess_lip / 2
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        if self.adaptive and self.hess is not None and not self.use_line_search:
            self.hess_lip /= 2
            empirical_lip = empirical_hess_lip(self.grad, self.grad_old, self.hess, self.x, self.x_old, self.loss)
            self.hess_lip = max(self.hess_lip, empirical_lip)
        if self.adaptive and not self.use_line_search:
            self.reg_coef = self.hess_lip / 2
        self.hess = self.loss.hessian(self.x)
        if self.use_line_search:
            self.x, self.reg_coef = self.line_search(self.x, self.reg_coef, self.grad, self.hess)
        else:
            grad_norm = self.loss.norm(self.grad)
            identity_coef = (self.reg_coef * grad_norm)**self.grad_norm_power
            self.x_old = copy.deepcopy(self.x)
            self.grad_old = copy.deepcopy(self.grad)
            # delta_x = -la.lstsq(self.hess + identity_coef*np.eye(self.loss.dim), self.grad, rcond=None)[0]
            delta_x = -np.linalg.solve(self.hess + identity_coef*np.eye(self.loss.dim), self.grad)
            self.x += delta_x
        
    def init_run(self, *args, **kwargs):
        super(RegNewton, self).init_run(*args, **kwargs)
        self.x_old = None
        self.hess = None
        self.trace.lrs = []
        
    def update_trace(self, *args, **kwargs):
        super(RegNewton, self).update_trace(*args, **kwargs)
        if self.adaptive and not self.use_line_search:
            self.trace.lrs.append(1 / self.reg_coef)
