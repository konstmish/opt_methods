from optimizer import Optimizer


class Bfgs(Optimizer):
    """
    Broyden–Fletcher–Goldfarb–Shanno algorithm.
    
    Arguments:
        L (float, optional): an upper bound on the smoothness constant
            to initialize the Hessian estimate
        hess_estim (float array of shape (dim, dim)): initial Hessian estimate
        lr (float, optional): stepsize (default: 1)
    """
    def __init__(self, L=None, hess_estim=None, lr=1, line_search=False, ls_backtrack=0.5, ls_it_max=100, *args, **kwargs):
        super(Bfgs, self).__init__(*args, **kwargs)
        if L is None and hess_estim is None:
            raise ValueError("Either smoothness consatnt L or Hessian estimate must be provided")
        self.lr = lr
        self.L = L
        self.B = hess_estim
        self.line_search = line_search
        self.ls_backtrack = ls_backtrack
        self.ls_it_max = ls_it_max
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        x_new = self.x - self.B_inv @ self.grad
        if self.line_search:
            lr = min(1., self.lrs[-1] / self.ls_backtrack if len(self.lrs) > 0 else .1)
            for i in range(self.ls_it_max):
                f_new = self.loss_func(self.w + lr * (w_new - self.w))
                if f_new < self.f:
                    break
                lr *= self.ls_backtrack
            self.lrs.append(lr)
            self.f = f_new
        else:
            lr = self.lr
        x_new = self.x + lr * (x_new - self.x)
        
        s = x_new - self.x
        grad_new = self.grad_func(x_new)
        y = grad_new - self.grad
        self.grad = grad_new
        Bs = self.B @ s
        sBs = s @ Bs
        Binv_y = self.B_inv @ y
        y_Binv_y = y @ Binv_y
        self.B += np.outer(y, y)/(y@s) - np.outer(Bs, Bs)/sBs
        self.B_inv += (s@y + y_Binv_y) * np.outer(s, s) / (s@y)**2
        self.B_inv -= (np.outer(Binv_y, s) + np.outer(s, Binv_y)) / (s@y)
        self.x = x_new
    
    def init_run(self, *args, **kwargs):
        super(Bfgs, self).init_run(*args, **kwargs)
        if not self.B:
            self.B = self.L * np.eye(self.loss.dim)
        self.B_inv = 1 / self.L * np.eye(len(self.w))
        self.grad = self.grad_func(self.w)
        if self.line_search:
            self.f = self.loss_func(self.w)
            self.lrs = []
