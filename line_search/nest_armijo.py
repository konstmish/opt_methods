import numpy as np

from .line_search import LineSearch

class NestArmijo(LineSearch):
    """
    This line search implements procedure (4.9) from the following paper by Nesterov:
        http://www.optimization-online.org/DB_FILE/2007/09/1784.pdf
    Arguments:
        mu (float, optional): strong convexity constant (default: 0.0)
        start_with_prev_lr (boolean, optional): initialize lr with the previous value (default: True)
        backtracking (float, optional): constant by which the current stepsize is multiplied (default: 0.5)
        start_with_small_momentum (bool, optional): momentum gradually increases. 
            Only used if mu>0 (default: True)
    """
    
    def __init__(self, mu=0, start_with_prev_lr=True, backtracking=0.5, start_with_small_momentum=True, *args, **kwargs):
        super(NestArmijo, self).__init__(count_first_it=True, *args, **kwargs)
        self.mu = mu
        self.start_with_prev_lr = start_with_prev_lr
        self.backtracking = backtracking
        self.start_with_small_momentum = start_with_small_momentum
        self.global_calls = 0
        
    def condition(self, y, x_new):
        grad_new = self.loss.gradient(x_new)
        return self.loss.inner_prod(grad_new, y - x_new) >= self.lr * self.loss.norm(grad_new)**2 - self.tolerance
        
    def __call__(self, x, v, A):
        self.global_calls += 1
        self.lr = self.lr / self.backtracking if self.start_with_prev_lr else self.lr0
        # Find $a$ from quadratic equation a^2/(A+a) = 2*lr*(1 + mu*A)
        discriminant = (self.lr * (1+self.mu*A)) ** 2 + A * self.lr * (1 + self.mu*A)
        a = self.lr * (1+self.mu*A) + np.sqrt(discriminant)
        if self.start_with_small_momentum:
            a_small = self.lr + np.sqrt(self.lr**2 + A * self.lr)
            a = min(a, a_small)
        y = (A*x + a*v) / (A+a)
        gradient = self.loss.gradient(y)
        x_new = y - self.lr * gradient
        nest_condition_met = self.condition(y, x_new)
        
        it_extra = 0
        it_max = min(2 * self.it_max, self.optimizer.ls_it_max - self.it)
        while not nest_condition_met and it_extra < it_max:
            self.lr *= self.backtracking
            discriminant = (self.lr * (1+self.mu*A)) ** 2 + A * self.lr * (1+self.mu*A)
            a = self.lr * (1+self.mu*A) + np.sqrt(discriminant)
            if self.start_with_small_momentum:
                a_small = self.lr + np.sqrt(self.lr**2 + A * self.lr)
                a = min(a, a_small)
            y = A / (A+a) * x + a / (A+a) *v
            gradient = self.loss.gradient(y)
            x_new = y - self.lr * gradient
            nest_condition_met = self.condition(y, x_new)
            it_extra += 2
            if self.lr * self.backtracking == 0:
                break
        
        self.it += self.it_per_call + it_extra
        return x_new, a
    
    def reset(self, *args, **kwargs):
        super(NestArmijo, self).reset(*args, **kwargs)
        self.global_calls = 0
