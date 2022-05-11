class LineSearch():
    """
    A universal Line Search class that allows for finding the best 
    scalar alpha such that x + alpha * delta is a good
    direction for optimization. The goodness of the new point can 
    be measured in many ways: decrease of functional values, 
    smaller gradient norm, Lipschitzness of an operator, etc.
    Arguments:
        lr0 (float, optional): the initial estimate (default: 1.0)
        count_first_it (bool, optional): to count the first iteration as requiring effort.
            This should be False for methods that reuse information, such as objective value, from the previous
            line search iteration. In contrast, most stochastic line searches
            should count the initial iteration too as information can't be reused (default: False)
        count_last_it (bool, optional): to count the last iteration as requiring effort.
            Not true for line searches that can use the produced information, such as matrix-vector
            product, to compute the next gradient or other important quantities. However, even then, 
            it is convenient to set to False to account for gradient computation (default: True)
        it_max (int, optional): maximal number of innert iterations per one call. 
            Prevents the line search from running for too long and from
            running into machine precision issues (default: 50)
        tolerance (float, optional): the allowed amount of condition violation (default: 0)
    """
    
    def __init__(self, lr0=1.0, count_first_it=False, count_last_it=True, it_max=50, tolerance=0):
        self.lr0 = lr0
        self.lr = lr0
        self.count_first_it = count_first_it
        self.count_last_it = count_last_it
        self.it = 0
        self.it_max = it_max
        self.tolerance = tolerance
        
    @property
    def it_per_call(self):
        return self.count_first_it + self.count_last_it
        
    def reset(self, optimizer):
        self.lr = self.lr0
        self.it = 0
        self.optimizer = optimizer
        self.loss = optimizer.loss
        self.use_prox = optimizer.use_prox
        
    def __call__(self, x=None, direction=None, x_new=None):
        pass
