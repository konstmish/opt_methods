import numpy as np
import numpy.linalg as la

from optmethods.optimizer import Optimizer


def ls_cubic_solver(x, g, H, M, it_max=100, epsilon=1e-8, loss=None):
    """
    Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3
    
    For explanation of Cauchy point, see "Gradient Descent 
        Efficiently Finds the Cubic-Regularized Non-Convex Newton Step"
        https://arxiv.org/pdf/1612.00547.pdf
    Other potential implementations can be found in paper
        "Adaptive cubic regularisation methods"
        https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf
    """
    solver_it = 1
    newton_step = -np.linalg.solve(H, g)
    if M == 0:
        return x + newton_step, solver_it
    def cauchy_point(g, H, M):
        if la.norm(g) == 0 or M == 0:
            return 0 * g
        g_dir = g / la.norm(g)
        H_g_g = H @ g_dir @ g_dir
        R = -H_g_g / (2*M) + np.sqrt((H_g_g/M)**2/4 + la.norm(g)/M)
        return -R * g_dir
    
    def conv_criterion(s, r):
        """
        The convergence criterion is an increasing and concave function in r
        and it is equal to 0 only if r is the solution to the cubic problem
        """
        s_norm = la.norm(s)
        return 1/s_norm - 1/r
    
    # Solution s satisfies ||s|| >= Cauchy_radius
    r_min = la.norm(cauchy_point(g, H, M))
    
    if loss is not None:
        x_new = x + newton_step
        if loss.value(x) > loss.value(x_new):
            return x_new, solver_it
        
    r_max = la.norm(newton_step)
    if r_max - r_min < epsilon:
        return x + newton_step, solver_it
    id_matrix = np.eye(len(g))
    for _ in range(it_max):
        # run bisection on the regularization using conv_criterion
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if np.abs(crit) < epsilon:
            return x + s_lam, solver_it
        if crit < 0:
            r_min = r_try
        else:
            r_max = r_try
        if r_max - r_min < epsilon:
            break
    return x + s_lam, solver_it


class Arc(Optimizer):
    """
    Adaptive Regularisation algorithm using Cubics (ARC) is a second-order optimizer based on Cubic Newton.
    This implementation is based on the paper by Cartis et al.,
        "Adaptive cubic regularisation methods for unconstrained optimization. 
        Part I: motivation, convergence and numerical results"
    We use the same rules for initializing eta1, eta2, sigma and updating sigma as given in the paper.
    
    Arguments:
        eta1 (float, optional): parameter to identify very successful iterations (default: 0.1)
        eta2 (float, optional): parameter to identify unsuccessful iterations (default: 0.9)
        eps (float, optional): minimal value of the cubic-penalty coefficient (default: 1e-16)
        sigma (float, optional): an estimate of the Hessian's Lipschitz constant
    """
    def __init__(self, eta1=0.1, eta2=0.9, eps=1e-16, sigma=1, solver_it=100, solver_eps=1e-8, cubic_solver=None, *args, **kwargs):
        super(Arc, self).__init__(*args, **kwargs)
        self.eta1 = eta1
        self.eta2 = eta2
        self.eps = eps
        self.sigma = sigma
        self.cubic_solver = cubic_solver
        self.solver_it = 0
        self.solver_it = solver_it
        self.solver_eps = solver_eps
        if sigma is None:
            self.sigma = self.loss.hessian_lipschitz
        if cubic_solver is None:
            self.cubic_solver = ls_cubic_solver
        self.f_prev = None
        self.line_search = 
        
    def step(self):
        if self.f_prev is None:
            self.f_prev = self.loss.value(self.x)
        self.grad = self.loss.gradient(self.x)
        grad_norm = self.loss.norm(self.grad)
        self.hess = self.loss.hessian(self.x)
        x_cubic, solver_it = self.cubic_solver(self.x, self.grad, self.hess, self.sigma, self.solver_it, self.solver_eps)
        s = x_cubic - self.x
        model_value = self.f_prev + self.loss.inner_prod(s, self.grad) + 0.5 * self.hess @ s @ s + self.sigma/3 * self.loss.norm(s)**3
        f_new = self.loss.value(x_cubic)
        rho = (self.f_prev - f_new) / (self.f_prev - model_value)
        if rho > self.eta1:
            self.x = x_cubic
            self.f_prev = f_new
        else:
            self.sigma *= 2
        if rho > self.eta2:
            self.sigma = max(self.eps, min(self.sigma, grad_norm))
        self.solver_it += solver_it
        
    def init_run(self, *args, **kwargs):
        super(Arc, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        
    def update_trace(self):
        super(Arc, self).update_trace()
        self.trace.solver_its.append(self.solver_it)
