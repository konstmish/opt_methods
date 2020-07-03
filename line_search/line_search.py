class LineSearch():
    """
    A universal Line Search class that allows for finding the best 
    scalar alpha such that x + alpha * delta is a good
    direction for optimization. The goodness of the new point can 
    be measured in many ways: decrease of functional values, 
    smaller gradient, Lipschitzness of an operator, etc.
    Args:
        count_first_it (bool, optional): to count the first iteration
            as if it takes effort. This should be False for methods that
            reuse information, such as objective value, from the previous 
            line search iteration. In contrast, most stochastic line searches
            should count the initial iteration too as it can't be reused (default: False)
        count_last_it (bool, optional): to count the last iteration as if
            it takes effort. Not true for line searches that can use the
            produced information, such as matrix-vector product, to compute
            the next gradient or other important quantities. However, even then, 
            it is convenient to set to False to account for gradient
            computation (default: True)
    """
    
    def __init__(self, lr0=1, count_first_it=False, count_last_it=True):
        self.lr0 = lr0
        self.lr = lr0
        self.count_first_it = count_first_it
        self.count_last_it = count_last_it
        self.it = 0
        
    @property
    def it_per_call(self):
        return self.count_first_it + self.count_last_it
        
    def reset(self, optimizer):
        self.lr = self.lr0
        self.it = 0
        self.optimizer = optimizer
        self.loss = optimizer.loss
        
    def __call__(self, direction=None, x=None, x_new=None):
        pass
