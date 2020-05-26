import numpy.linalg as la


from optimizer import Optimizer


class Newton(Optimizer):
    """
    Gradient descent with constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=1, *args, **kwargs):
        super(Newton, self).__init__(*args, **kwargs)
        self.lr = lr
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        self.hess = self.loss.hessian(self.x)
        inv_hess_grad_prod = la.lstsq(self.hess, self.grad, rcond=None)[0]
        self.x -= self.lr * inv_hess_grad_prod
