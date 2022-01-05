import numpy.linalg as la

from opt_methods.optimizer import Optimizer


class Newton(Optimizer):
    """
    Newton algorithm for convex minimization.
    
    Arguments:
        lr (float, optional): dampening constant (default: 1)
    """
    def __init__(self, lr=1, *args, **kwargs):
        super(Newton, self).__init__(*args, **kwargs)
        self.lr = lr
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.hess = self.loss.hessian(self.x)
        inv_hess_grad_prod = la.lstsq(self.hess, self.grad, rcond=None)[0]
        if self.line_search is None:
            self.x -= self.lr * inv_hess_grad_prod
        else:
            self.x = self.line_search(x=self.x, direction=-inv_hess_grad_prod)
