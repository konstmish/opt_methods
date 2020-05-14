from optimizer import Optimizer


class Nesterov(Optimizer):
    """
    Gradient descent with constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=None, strongly_convex=False, mu=0, *args, **kwargs):
        super(Nesterov, self).__init__(*args, **kwargs)
        self.lr = lr
        if mu < 0:
            raise ValueError("Invalid mu: {}".format(mu))
        if strongly_convex and mu == 0:
            raise ValueError("""Mu must be larger than 0 for strongly_convex=True,
                             invalid value: {}""".format(mu))
        if strongly_convex:
            self.mu = mu
            kappa = (1/self.lr)/self.mu
            self.momentum = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)
        self.strongly_convex = strongly_convex
        
    def step(self):
        if not self.strongly_convex:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4*self.alpha**2))
            self.momentum = (self.alpha - 1) / alpha_new
            self.alpha = alpha_new
        self.x_nest_old = self.x_nest.copy()
        self.grad = self.loss.gradient(self.x)
        self.x_nest = self.x - self.lr*self.grad
        self.x = self.x_nest + momentum*(self.x_nest-self.x_nest_old)
    
    def init_run(self, *args, **kwargs):
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness()
        super(Nesterov, self).init_run(*args, **kwargs)
        self.x_nest = self.x.copy()
        self.alpha = 1.