import copy
import numpy as np

from optimizer import Optimizer


class Lbfgs(Optimizer):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm. See
        p. 177 in J. Nocedal S. J. Wright, "Numerical Optimization", 2nd edtion
    or 
        https://en.wikipedia.org/wiki/Limited-memory_BFGS
    for a general description.
    
    Arguments:
        L (float, optional): an upper bound on the smoothness constant
            to initialize the Hessian estimate
        hess_estim (float array of shape (dim, dim), optional): initial Hessian estimate
        lr (float, optional): stepsize (default: 1)
        mem_size (int, optional): memory size (default: 1)
        lm (float, optional): Levenberg-Marquardt penalty (default: 0)
    """
    
    def __init__(self, L=None, hess_estim=None, inv_hess_estim=None, lr=1, mem_size=1, adaptive_init=False, *args, **kwargs):
        super(Lbfgs, self).__init__(*args, **kwargs)
        if L is None and hess_estim is None and inv_hess_estim is None:
            L = self.loss.smoothness()
            if L is None:
                raise ValueError("Either smoothness constant L or Hessian/inverse-Hessian estimate must be provided")
        self.L = L
        self.lr = lr
        self.mem_size = mem_size
        self.adaptive_init = adaptive_init
        self.B = hess_estim
        self.B_inv = inv_hess_estim
        if inv_hess_estim is None and hess_estim is not None:
            self.B_inv = np.linalg.pinv(self.B)
        self.x_difs = []
        self.grad_difs = []
        self.rhos = []
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        q = copy.deepcopy(self.grad)
        alphas = []
        for s, y, rho in zip(reversed(self.x_difs), reversed(self.grad_difs), reversed(self.rhos)):
            alpha = rho * self.loss.inner_prod(s, q)
            alphas.append(alpha)
            q -= alpha * y
        if self.B_inv is not None:
            r = self.B_inv @ q
        else:
            if self.adaptive_init and len(self.x_difs) > 0:
                y = self.grad_difs[-1]
                y_norm = self.loss.norm(y)
                self.L_local = y_norm**2 * self.rhos[-1]
                r = q / self.L_local
            else:
                r = q / self.L
        for s, y, rho, alpha in zip(self.x_difs, self.grad_difs, self.rhos, reversed(alphas)):
            beta = rho * self.loss.inner_prod(y, r)
            r += s * (alpha - beta)
        
        x_new = self.x - self.lr * r
        if self.line_search is not None:
            x_new = self.line_search(self.x, x_new)
        grad_new = self.loss.gradient(x_new)
        
        rho_inv = self.loss.inner_prod(grad_new - self.grad, x_new - self.x)
        if rho_inv > 0:
            self.x_difs.append(x_new - self.x)
            self.grad_difs.append(grad_new - self.grad)
            self.rhos.append(1 / rho_inv)
            
        if len(self.x_difs) > self.mem_size:
            self.x_difs.pop(0)
            self.grad_difs.pop(0)
            self.rhos.pop(0)
        self.x = x_new
        self.grad = grad_new
        
#         s = x_new - self.x
#         grad_new = self.loss.gradient(x_new)
#         y = grad_new - self.grad
#         self.grad = grad_new
#         Bs = self.B @ s
#         sBs = s @ Bs
#         B_inv_y = self.B_inv @ y
#         y_B_inv_y = y @ B_inv_y
#         y_s = y @ s
#         self.B += np.outer(y, y)/y_s - np.outer(Bs, Bs)/sBs
#         self.B_inv += (y_s + y_B_inv_y) * np.outer(s, s) / y_s**2
#         self.B_inv -= (np.outer(B_inv_y, s) + np.outer(s, B_inv_y)) / y_s
#         self.x = x_new
    
#     def init_run(self, *args, **kwargs):
#         super(Lbfgs, self).init_run(*args, **kwargs)
#         self.grad = self.loss.gradient(self.x)
