import copy

from .line_search import LineSearch

class Armijo(LineSearch):
    """
    Armijo line search with optional resetting of the initial stepsize
    of each iteration. If resetting is used, the previous value multiplied 
    by 1/backtracking is used as the first stepsize to try at this iteration.
    Otherwise, it starts with the maximal stepsize.
    Arguments:
        armijo_const (float, optional): proportionality constant for the armijo condition (default: 0.5)
        start_with_prev_lr (boolean, optional): sets the reset option from (default: True)
        backtracking (float, optional): constant to multiply the estimate stepsize with (default: 0.5)
    """
    
    def __init__(self, armijo_const=0.5, start_with_prev_lr=True, backtracking=0.5, *args, **kwargs):
        super(Armijo, self).__init__(*args, **kwargs)
        self.armijo_const = armijo_const
        self.start_with_prev_lr = start_with_prev_lr
        self.backtracking = backtracking
        self.x_prev = None
        self.val_prev = None
        
    def condition(self, gradient, x, x_new):
        value_new = self.loss.value(x_new)
        self.x_prev = copy.deepcopy(x_new)
        self.val_prev = value_new
        descent = self.armijo_const * self.loss.inner_prod(gradient, x - x_new)
        return value_new <= self.current_value - descent
        
    def __call__(self, gradient=None, direction=None, x=None, x_new=None):
        if gradient is None:
            gradient = self.optimizer.grad
        if x is None:
            x = self.optimizer.x
        self.lr = self.lr / self.backtracking if self.start_with_prev_lr else self.lr0
        if direction is None:
            direction = (x_new - x) / self.lr
        if x_new is None:
            x_new = x + direction * self.lr
        if x is self.x_prev:
            self.current_value = self.val_prev
        else:
            self.current_value = self.loss.value(x)
        
        armijo_condition_met = self.condition(gradient, x, x_new)
        it_extra = 0
        while not armijo_condition_met:
            self.lr *= self.backtracking
            x_new = x + self.lr * direction
            armijo_condition_met = self.condition(gradient, x, x_new)
            it_extra += 1
        
        self.it += self.it_per_call + it_extra
        return x_new
