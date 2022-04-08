import numpy as np

import numpy.linalg as la

from .line_search import LineSearch

class RegNewtonLS(LineSearch):
    """
    This line search estimates the Hessian Lipschitz constant for the Global Regularized Newton.
    See the following paper for the details and convergence proof:
        "Regularized Newton Method with Global O(1/k^2) Convergence"
        https://arxiv.org/abs/2112.02089
    For consistency with other line searches, 'lr' parameter is used to denote the inverse of regularization.
    Arguments:
        decrease_reg (boolean, optional): multiply the previous regularization parameter by 1/backtracking (default: True)
        backtracking (float, optional): constant by which the current regularization is divided (default: 0.5)
    """
    
    def __init__(self, decrease_reg=True, backtracking=0.5, H0=None, *args, **kwargs):
        super(RegNewtonLS, self).__init__(*args, **kwargs)
        self.decrease_reg = decrease_reg
        self.backtracking = backtracking
        self.H0 = H0
        self.H = self.H0
        self.attempts = 0
        
    def condition(self, x_new, x, grad, identity_coef):
        if self.f_prev is None:
            self.f_prev = self.loss.value(x)
        self.f_new = self.loss.value(x_new)
        r = self.loss.norm(x_new - x)
        condition_f = self.f_new <= self.f_prev - 2/3 * identity_coef * r**2
        grad_new = self.loss.gradient(x_new)
        condition_grad = self.loss.norm(grad_new) <= 2 * identity_coef * r
        self.attempts = self.attempts + 1 if not condition_f or not condition_grad else 0
        return condition_f and condition_grad
        
    def __call__(self, x, grad, hess):
        if self.decrease_reg:
            self.H *= self.backtracking
        grad_norm = self.loss.norm(grad)
        identity_coef = np.sqrt(self.H * grad_norm)
        
        x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
        condition_met = self.condition(x_new, x, grad, identity_coef)
        self.it += self.it_per_call
        it_extra = 0
        it_max = min(self.it_max, self.optimizer.ls_it_max - self.it)
        while not condition_met and it_extra < it_max:
            self.H /= self.backtracking
            identity_coef = np.sqrt(self.H * grad_norm)
            x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
            condition_met = self.condition(x_new, x, grad, identity_coef)
            it_extra += 1
            if self.backtracking / self.H == 0:
                break
        self.f_prev = self.f_new
        self.it += it_extra
        self.lr = 1 / identity_coef
        return x_new

    def reset(self, *args, **kwargs):
        super(RegNewtonLS, self).reset(*args, **kwargs)
        self.f_prev = None
