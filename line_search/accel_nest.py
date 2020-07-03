import numpy as np

from .line_search import LineSearch

class AccelNest(LineSearch):
    """
    This line search implements procedure (4.9) from the following paper by Nesterov:
    http://www.optimization-online.org/DB_FILE/2007/09/1784.pdf
    Arguments:
        mu (float, optional): strong convexity constant (default: 0.0)
        start_with_prev_lr (boolean, optional): sets the reset option from (default: True)
        backtracking (float, optional): constant to multiply the estimate stepsize with (default: 0.5)
    """
    
    def __init__(self, mu=0, start_with_prev_lr=True, backtracking=0.5, *args, **kwargs):
        super(AccelNest, self).__init__(count_first_it=True, *args, **kwargs)
        self.mu = mu
        self.start_with_prev_lr = start_with_prev_lr
        self.backtracking = backtracking
        
    def condition(self, y, x_new):
        grad_new = self.loss.gradient(x_new)
        return self.loss.inner_prod(grad_new, y - x_new) >= self.lr * self.loss.norm(grad_new)**2
        
    def __call__(self, x, v, A):
        self.lr = self.lr / self.backtracking if self.start_with_prev_lr else self.lr0
        # Find a from quadratic equation a^2/(A+a) = 2*lr*(1 + mu*A)
        discriminant = (self.lr * (1 + self.mu*A)) ** 2 + A * self.lr * (1 + self.mu*A)
        a = self.lr * (1 + self.mu*A) + np.sqrt(discriminant)
        y = (A*x + a*v) / (A+a)
        gradient = self.loss.gradient(y)
        x_new = y - self.lr * gradient
        nest_condition_met = self.condition(y, x_new)
        
        it_extra = 0
        while not nest_condition_met:
            self.lr *= self.backtracking
            discriminant = (self.lr * (1 + self.mu*A)) ** 2 + A * self.lr * (1 + self.mu*A)
            a = self.lr * (1 + self.mu*A) + np.sqrt(discriminant)
            y = A / (A + a) * x + a / (A + a) *v
            gradient = self.loss.gradient(y)
            x_new = y - self.lr * gradient
            nest_condition_met = self.condition(y, x_new)
            it_extra += 2
            if self.lr * self.backtracking == 0:
                break
        
        self.it += self.it_per_call + it_extra
        return x_new, a
