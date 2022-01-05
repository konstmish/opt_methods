from opt_methods.optimizer import Optimizer


class HeavyBall(Optimizer):
    """
    Gradient descent with Polyak's heavy-ball momentum
    For details, see, e.g., https://vsokolov.org/courses/750/files/polyak64.pdf
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        momentum (float, optional): momentum value. For quadratics, 
            it should be close to 1-sqrt(l_min/l_max), where l_min and
            l_max are the smallest/largest eigenvalues of the quadratic matrix
    """
    def __init__(self, lr=None, strongly_convex=False, momentum=None, *args, **kwargs):
        super(HeavyBall, self).__init__(*args, **kwargs)
        self.lr = lr
        if momentum < 0:
            raise ValueError("Invalid momentum: {}".format(mu))
        self.strongly_convex = strongly_convex
        if self.strongly_convex:
            self.momentum = momentum
        
    def step(self):
        if not self.strongly_convex:
            self.momentum = self.it / (self.it+1)
        x_copy = self.x.copy()
        self.grad = self.loss.gradient(self.x)
        if self.use_prox:
            self.x = self.loss.regularizer.prox(self.x - self.lr*self.grad, self.lr) + self.momentum*(self.x-self.x_old)
        else:
            self.x = self.x - self.lr * self.grad + self.momentum*(self.x-self.x_old)
        self.x_old = x_copy
    
    def init_run(self, *args, **kwargs):
        super(HeavyBall, self).init_run(*args, **kwargs)
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness
        self.x_old = self.x.copy()
