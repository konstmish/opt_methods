import numpy as np

from optimizer import Optimizer


class Sr1(Optimizer):
    """
    Quasi-Newton algorithm with Symmetric Rank 1 (SR1) update. See 
    https://arxiv.org/pdf/2002.00657.pdf
    for a formal description and convergence proof of a similar method.
    
    Arguments:
        L (float, optional): an upper bound on the smoothness constant
            to initialize the Hessian estimate
        hess_estim (float array of shape (dim, dim)): initial Hessian estimate
        lr (float, optional): stepsize (default: 1)
        stability_const (float, optional): a constant from [0, 1) that ensures a curvature
            condition before updating the Hessian-inverse estimate (default: 0.)
    """
    
    def __init__(self, L=None, hess_estim=None, lr=1, stability_const=0., *args, **kwargs):
        super(Sr1, self).__init__(*args, **kwargs)
        if L is None and hess_estim is None:
            L = self.loss.smoothness()
            if L is None:
                raise ValueError("Either smoothness constant L or Hessian estimate must be provided")
        if not 0 <= stability_const < 1:
            raise ValueError("Invalid stability parameter: {}".format(stability_const))
        self.lr = lr
        self.L = L
        self.B = hess_estim
        self.stability_const = stability_const
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        x_new = self.x - self.lr * self.B_inv @ self.grad
        if self.line_search is not None:
            x_new = self.line_search(self.x, x_new)
        
        s = x_new - self.x
        grad_new = self.loss.gradient(x_new)
        y = grad_new - self.grad
        self.grad = grad_new
        Bs = self.B @ s
        sBs = s @ Bs
        B_inv_y = self.B_inv @ y
        y_B_inv_y = y @ B_inv_y
        y_s = y @ s
        if abs(y_s-sBs) > self.stability_const * self.loss.norm(s) * self.loss.norm(y-Bs):
            self.B += np.outer(y-Bs, y-Bs) / (y_s-sBs)
            self.B_inv += np.outer(s-B_inv_y, s-B_inv_y) / (y_s-y_B_inv_y)
        self.x = x_new
    
    def init_run(self, *args, **kwargs):
        super(Sr1, self).init_run(*args, **kwargs)
        if self.B is None:
            self.B = self.L * np.eye(self.loss.dim)
            self.B_inv = 1 / self.L * np.eye(self.loss.dim)
        else:
            self.B_inv = np.linalg.pinv(self.B)
        self.grad = self.loss.gradient(self.x)
