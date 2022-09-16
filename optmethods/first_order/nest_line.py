import copy

from optmethods.line_search import NestArmijo
from optmethods.optimizer import Optimizer


class NestLine(Optimizer):
    """
    Accelerated gradient descent with line search proposed by Nesterov.
    For details, see, equation (4.9) in
        http://www.optimization-online.org/DB_FILE/2007/09/1784.pdf
    The method does not support increasing momentum, which may limit 
    its efficiency on ill-conditioned problems.
    
    Arguments:
        line_search (optmethods.LineSearch, optional): a callable line search, here it should be None or
            an instance of NestArmijo class.  If None, line search is intialized automatically (default: None)
        lr (float, optional): an estimate of the inverse smoothness constant
        strongly_convex (bool, optional): use the variant for strongly convex functions,
            which requires mu to be provided (default: False)
        mu (float, optional): strong-convexity constant or a lower bound on it (default: 0)
        start_with_small_momentum (bool, optional): momentum gradually increases. 
            Only used if mu>0 (default: True)
    """
    def __init__(self, line_search=None, lr=None, mu=0, start_with_small_momentum=True, *args, **kwargs):
        if mu < 0:
            raise ValueError("Invalid mu: {}".format(mu))
        if line_search is None:
            line_search = NestArmijo(mu=mu, start_with_small_momentum=start_with_small_momentum)
        super(NestLine, self).__init__(line_search=line_search, *args, **kwargs)
        self.lr = lr
        self.mu = mu
        self.start_with_small_momentum = start_with_small_momentum
        
    def step(self):
        self.x, a = self.line_search(self.x, self.v, self.A)
        self.A += a
        self.grad = self.loss.gradient(self.x)
        self.v -= a * self.grad
    
    def init_run(self, *args, **kwargs):
        super(NestLine, self).init_run(*args, **kwargs)
        self.v = copy.deepcopy(self.x)
        self.A = 0
