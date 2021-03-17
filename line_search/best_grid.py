import copy
import numpy as np

from .line_search import LineSearch

class BestGrid(LineSearch):
    """
    Find the best stepsize either over values {lr_max * backtracking ** pow: pow=0, 1, ...} or
        over {lr * backtracking ** (1 - pow): pow=0, 1, ...} where lr is the previous value
    Arguments:
        lr_max (float, optional): the maximal stepsize, useful for second-order 
            and quasi-Newton methods (default: np.inf)
        functional (boolean, optional): use functional values to check optimality. 
            Otherwise, gradient norm is used (default: True)
        start_with_prev_lr (boolean, optional): initialize lr with the previous value (default: True)
        increase_lr (boolean, optional): multiply the previous lr by 1/backtracking (default: True)
        increase_many_times (boolean, optional): multiply the lr by 1/backtracking until it's the best (default: False)
        backtracking (float, optional): constant to multiply the estimated stepsize by (default: 0.5)
    """
    
    def __init__(self, lr_max=np.inf, functional=True, start_with_prev_lr=False, increase_lr=True,
                 increase_many_times=True, backtracking=0.5, *args, **kwargs):
        super(BestGrid, self).__init__(*args, **kwargs)
        self.lr_max = lr_max
        self.functional = functional
        self.start_with_prev_lr = start_with_prev_lr
        self.increase_lr = increase_lr
        self.increase_many_times = increase_many_times
        self.backtracking = backtracking
        self.x_prev = None
        self.val_prev = None
        
    def condition(self, proposed_value, proposed_next):
        return proposed_value <= proposed_next + self.tolerance
    
    def metric_value(self, x):
        if self.functional:
            return self.loss.value(x)
        return self.loss.norm(self.loss.gradient(x))
        
    def __call__(self, x, x_new=None, direction=None):
        if x is None:
            x = self.optimizer.x
        if direction is None:
            direction = x_new - x
            self.lr = 1
        elif self.start_with_prev_lr:
            self.lr = self.lr / self.backtracking if self.increase_lr else self.lr
            self.lr = min(self.lr, self.lr_max)
        else:
            self.lr = self.lr0
        if x_new is None:
            x_new = x + self.lr * direction
        if self.loss.is_equal(x, self.x_prev):
            self.current_value = self.val_prev
        else:
            self.current_value = self.metric_value(x)
        
        it_extra = 0
        proposed_value = self.metric_value(x_new)
        need_to_decrease_lr = proposed_value > self.current_value
        if not need_to_decrease_lr:
            x_next = x + self.lr * self.backtracking * direction
            proposed_next = self.metric_value(x_next)
            if not self.condition(proposed_value, proposed_next):
                need_to_decrease_lr = True
                self.lr *= self.backtracking
                proposed_value = proposed_next
            it_extra += 1
        found_best = not need_to_decrease_lr and not self.increase_many_times
        while not found_best and it_extra < self.it_max:
            if need_to_decrease_lr:
                lr_next = self.lr * self.backtracking
            else:
                lr_next = min(self.lr / self.backtracking, self.lr_max)
            x_next = x + lr_next * direction
            proposed_next = self.metric_value(x_next)
            found_best = self.condition(proposed_value, proposed_next)
            it_extra += 1
            if not found_best or it_extra == self.it_max:
                self.lr = lr_next
                proposed_value = proposed_next
        
        x_new = x + self.lr * direction
        self.val_prev = proposed_value
        self.x_prev = copy.deepcopy(x_new)
        self.it += self.it_per_call + it_extra
        return x_new
