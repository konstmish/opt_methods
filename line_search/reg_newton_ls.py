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
        decrease_reg (boolean, optional): multiply the previous regularization by 1/backtracking (default: True)
        backtracking (float, optional): constant by which the current regularization is divided (default: 0.5)
    """
    
    def __init__(self, decrease_reg=True, backtracking=0.5, *args, **kwargs):
        super(RegNewtonLS, self).__init__(*args, **kwargs)
        self.decrease_reg = decrease_reg
        self.backtracking = backtracking
        self.attempts = 0
        
    def condition(self, x_new, x, grad, reg_coef):
        if self.f_prev is None:
            self.f_prev = self.loss.value(x)
        self.f_new = self.loss.value(x_new)
        r = self.loss.norm(x_new - x)
        condition_f = self.f_new <= self.f_prev - 2/3 * reg_coef * r**2
        grad_new = self.loss.gradient(x_new)
        condition_grad = self.loss.norm(grad_new) <= 2 * reg_coef * r
        # condition_grad = self.loss.norm(grad_new) <= 2 * self.loss.norm(grad)
        self.attempts = self.attempts + 1 if not condition_f or not condition_grad else 0
        return condition_f and condition_grad
        
    def __call__(self, x, reg_coef, grad, hess):
        if self.decrease_reg:
            reg_coef *= self.backtracking
        grad_norm = self.loss.norm(grad)
        identity_coef = np.sqrt(reg_coef * grad_norm)
        
        x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
        condition_met = self.condition(x_new, x, grad, identity_coef)
        self.it += self.it_per_call
        it_extra = 0
        it_max = min(self.it_max, self.optimizer.ls_it_max - self.it)
        while not condition_met and it_extra < it_max:
            reg_coef /= self.backtracking
            identity_coef = np.sqrt(reg_coef * grad_norm)
            # x_new = x - la.lstsq(hess + identity_coef*np.eye(self.loss.dim), grad, rcond=None)[0]
            x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
            condition_met = self.condition(x_new, x, grad, identity_coef)
            it_extra += 1
            if self.backtracking / reg_coef == 0:
                break
        self.f_prev = self.f_new
        self.it += it_extra
        self.lr = 1 / reg_coef
        return x_new, reg_coef

    def reset(self, *args, **kwargs):
        super(RegNewtonLS, self).reset(*args, **kwargs)
        self.f_prev = None
