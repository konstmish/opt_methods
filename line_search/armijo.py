import copy

from .line_search import LineSearch

class Armijo(LineSearch):
    """
    Armijo line search with optional resetting of the initial stepsize
    at each iteration. If resetting is used, the previous value is optionally 
    multiplied by 1/backtracking and used as the first stepsize to try at the
    new iteration. Otherwise, it starts with the maximal stepsize.
    Arguments:
        armijo_const (float, optional): proportionality constant for the armijo condition (default: 0.5)
        start_with_prev_lr (boolean, optional): initialize lr with the previous value (default: True)
        increase_lr (boolean, optional): multiply the previous lr by 1/backtracking (default: True)
        backtracking (float, optional): constant by which the current stepsize is multiplied (default: 0.5)
    """
    
    def __init__(self, armijo_const=0.5, start_with_prev_lr=True, increase_lr=True, backtracking=0.5, *args, **kwargs):
        super(Armijo, self).__init__(*args, **kwargs)
        self.armijo_const = armijo_const
        self.start_with_prev_lr = start_with_prev_lr
        self.increase_lr = increase_lr
        self.backtracking = backtracking
        self.x_prev = None
        self.val_prev = None
        
    def condition(self, gradient, x, x_new):
        value_new = self.loss.value(x_new)
        self.val_prev = value_new
        descent = self.armijo_const * self.loss.inner_prod(gradient, x - x_new)
        return value_new <= self.current_value - descent + self.tolerance
        
    def __call__(self, x=None, x_new=None, gradient=None, direction=None):
        if gradient is None:
            gradient = self.optimizer.grad
        if x is None:
            x = self.optimizer.x
        if direction is None:
            direction = (x_new - x) / self.lr
        if self.start_with_prev_lr:
            self.lr = self.lr / self.backtracking if self.increase_lr else self.lr
        else:
            self.lr = self.lr0
        if x_new is None:
            x_new = x + self.lr * direction
        if self.loss.is_equal(x, self.x_prev):
            self.current_value = self.val_prev
        else:
            self.current_value = self.loss.value(x)
        
        armijo_condition_met = self.condition(gradient, x, x_new)
        it_extra = 0
        while not armijo_condition_met and it_extra < self.it_max:
            self.lr *= self.backtracking
            x_new = x + self.lr * direction
            armijo_condition_met = self.condition(gradient, x, x_new)
            it_extra += 1
        
        self.x_prev = copy.deepcopy(x_new)
        self.it += self.it_per_call + it_extra
        return x_new
