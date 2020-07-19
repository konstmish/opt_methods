import copy

from .line_search import LineSearch

class Wolfe(LineSearch):
    """
    Wolfe line search with optional resetting of the initial stepsize
    at each iteration. If resetting is used, the previous value is used
    as the first stepsize to try at this iteration. Otherwise, it starts
    with the maximal stepsize.
    Arguments:
        armijo_const (float, optional): proportionality constant for the armijo condition (default: 0.5)
        wolfe_const (float, optional): second proportionality constant for the wolfe condition (default: 0.5)
        start_with_prev_lr (boolean, optional): sets the reset option from (default: True)
        backtracking (float, optional): constant by which the current stepsize is multiplied (default: 0.5)
    """
    
    def __init__(self, armijo_const=0.1, wolfe_const=0.9, strong=False, 
                 start_with_prev_lr=True, backtracking=0.5, *args, **kwargs):
        super(Wolfe, self).__init__(*args, **kwargs)
        self.armijo_const = armijo_const
        self.wolfe_const = wolfe_const
        self.strong = strong
        self.start_with_prev_lr = start_with_prev_lr
        self.backtracking = backtracking
        self.x_prev = None
        self.val_prev = None
    
    def armijo_condition(self, gradient, x, x_new):
        value_new = self.loss.value(x_new)
        self.x_prev = copy.deepcopy(x_new)
        self.val_prev = value_new
        descent = self.armijo_const * self.loss.inner_prod(gradient, x - x_new)
        return value_new <= self.current_value - descent + self.tolerance
    
    def curvature_condition(self, gradient, x, x_new):
        grad_new = self.loss.gradient(x_new)
        curv_x = self.loss.inner_prod(gradient, x - x_new)
        curv_x_new = self.loss.inner_prod(grad_new, x - x_new)
        if self.strong:
            curv_x, curv_x_new = np.abs(curv_x), np.abs(curv_x_new)
        return curv_x_new <= self.wolfe_const * curv_x + self.tolerance
        
    def __call__(self, x=None, x_new=None, gradient=None, direction=None):
        if gradient is None:
            gradient = self.optimizer.grad
        if x is None:
            x = self.optimizer.x
        if direction is None:
            direction = (x_new - x) / self.lr
        self.lr = self.lr if self.start_with_prev_lr else self.lr0
        if x_new is None:
            x_new = x + self.lr * direction
        if self.loss.is_equal(x, self.x_prev):
            self.current_value = self.val_prev
        else:
            self.current_value = self.loss.value(x)
        
        armijo_condition = self.armijo_condition(gradient, x, x_new)
        curvature_condition = self.curvature_condition(gradient, x, x_new)
        it_extra = 0
        while not armijo_condition and it_extra < self.it_max:
            self.lr *= self.backtracking
            x_new = x + self.lr * direction
            armijo_condition = self.armijo_condition(gradient, x, x_new)
            it_extra += 1
        if it_extra == 0:
            while not curvature_condition and it_extra < self.it_max:
                self.lr /= self.backtracking
                x_new = x + self.lr * direction
                curvature_condition = self.curvature_condition(gradient, x, x_new)
                it_extra += 1
        
        self.it += self.it_per_call + it_extra
        return x_new
