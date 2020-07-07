import copy

from .line_search import LineSearch

class Goldstein(LineSearch):
    """
    Goldstein line search with optional resetting of the initial stepsize
    at each iteration. Combines Armijo condition with an extra condition to make sure
    that the stepsize is not too small. If resetting is used, the previous value 
    is used as the first stepsize to try at this iteration. Otherwise,
    it starts with the maximal stepsize.
    Arguments:
        goldstein_const (float, optional): proportionality constant for both conditions (default: 0.05)
        start_with_prev_lr (boolean, optional): sets the reset option from (default: True)
        backtracking (float, optional): constant to multiply the estimate stepsize with (default: 0.5)
    """
    
    def __init__(self, goldstein_const=0.05, start_with_prev_lr=True, backtracking=0.5, *args, **kwargs):
        super(Goldstein, self).__init__(*args, **kwargs)
        self.goldstein_const = goldstein_const
        self.start_with_prev_lr = start_with_prev_lr
        self.backtracking = backtracking
        self.x_prev = None
        self.val_prev = None
        
    def armijo_condition(self, gradient, x, x_new):
        value_new = self.loss.value(x_new)
        self.x_prev = copy.deepcopy(x_new)
        self.val_prev = value_new
        descent = self.goldstein_const * self.loss.inner_prod(gradient, x - x_new)
        return value_new <= self.current_value - descent
    
    def goldstein_condition(self, gradient, x, x_new):
        value_new = self.loss.value(x_new)
        self.x_prev = copy.deepcopy(x_new)
        self.val_prev = value_new
        descent = (1-self.goldstein_const) * self.loss.inner_prod(gradient, x - x_new)
        return value_new >= self.current_value - descent
        
    def __call__(self, gradient=None, direction=None, x=None, x_new=None):
        if gradient is None:
            gradient = self.optimizer.grad
        if x is None:
            x = self.optimizer.x
        self.lr = self.lr if self.start_with_prev_lr else self.lr0
        if direction is None:
            direction = (x_new - x) / self.lr
        if x_new is None:
            x_new = x + direction * self.lr
        if x is self.x_prev:
            self.current_value = self.val_prev
        else:
            self.current_value = self.loss.value(x)
        
        armijo_condition = self.armijo_condition(gradient, x, x_new)
        goldstein_condition = self.goldstein_condition(gradient, x, x_new)
        it_extra = 0
        while not armijo_condition:
            self.lr *= self.backtracking
            x_new = x + self.lr * direction
            armijo_condition = self.armijo_condition(gradient, x, x_new)
            it_extra += 1
        if it_extra == 0:
            while not goldstein_condition:
                self.lr /= self.backtracking
                x_new = x + self.lr * direction
                goldstein_condition = self.goldstein_condition(gradient, x, x_new)
                it_extra += 1
        
        self.it += self.it_per_call + it_extra
        return x_new
