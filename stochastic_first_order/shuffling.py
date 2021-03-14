import math
import numpy as np

from optimizer import StochasticOptimizer


class Shuffling(StochasticOptimizer):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.
    For a formal description and convergence guarantees, see
        https://arxiv.org/abs/2006.05988
    
    The method is sensitive to finishing the final epoch, so it will terminate earlier 
    than it_max if it_max is not divisible by the number of steps per epoch.
    
    Arguments:
        reshuffle (bool, optional): whether to get a new permuation for every new epoch.
            For convex problems, only a single permutation should suffice and it can run faster (default: False)
        prox_every_it (bool, optional): whether to use proximal operation every iteration 
            or only at the end of an epoch. Theory supports the latter. Only used if the loss includes
            a proximal regularizer (default: False)
        lr0 (float, optional): an estimate of the inverse smoothness constant, this step-size
            is used for the first epoch_start_decay epochs. If not given, it will be set
            with the value in the loss.
        lr_max (float, optional): a maximal step-size never to be exceeded (default: np.inf)
        lr_decay_coef (float, optional): the coefficient in front of the number of finished epochs
            in the denominator of step-size. For strongly convex problems, a good value
            is mu/3, where mu is the strong convexity constant
        lr_decay_power (float, optional): the power to exponentiate the number of finished epochs
            in the denominator of step-size. For strongly convex problems, a good value is 1 (default: 1)
        epoch_start_decay (int, optional): how many epochs the step-size is kept constant
            By default, will be set to have about 2.5% of iterations with the step-size equal to lr0
        batch_size (int, optional): the number of samples from the function to be used at each iteration
        importance_sampling (bool, optional): use importance sampling to speed up convergence
        update_trace_at_epoch_end (bool, optional): save progress only at the end of an epoch, which 
            avoids bad iterates
    """
    def __init__(self, reshuffle=False, prox_every_it=False, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, epoch_start_decay=1, batch_size=1, importance_sampling=False, 
                 update_trace_at_epoch_end=True, *args, **kwargs):
        super(Shuffling, self).__init__(*args, **kwargs)
        self.reshuffle = reshuffle
        self.prox_every_it = prox_every_it
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.epoch_start_decay = epoch_start_decay
        self.batch_size = batch_size
        self.importance_sampling = importance_sampling
        self.update_trace_at_epoch_end = update_trace_at_epoch_end
        
        self.steps_per_epoch = math.ceil(self.loss.n/batch_size)
        self.epoch_max = self.it_max // self.steps_per_epoch
        if epoch_start_decay is None and np.isfinite(self.epoch_max):
            self.epoch_start_decay = 1 + self.epoch_max // 40
        elif epoch_start_decay is None:
            self.epoch_start_decay = 1
        if importance_sampling:
            self.sample_counts = self.loss.individ_smoothness / np.mean(self.loss.individ_smoothness)
            self.sample_counts = np.int64(np.ceil(self.sample_counts))
            self.idx_with_copies = np.repeat(np.arange(self.loss.n), self.sample_counts)
            self.n_copies = sum(self.sample_counts)
            self.steps_per_epoch = math.ceil(self.n_copies / batch_size)
        
    def step(self):
        if self.it%self.steps_per_epoch == 0:
            # Start new epoch
            if self.reshuffle:
                if not self.importance_sampling:
                    self.permutation = np.random.permutation(self.loss.n)
                else:
                    self.permutation = np.random.permutation(self.idx_with_copies)
                self.sampled_permutations += 1
            self.i = 0
        i_max = min(len(self.permutation), self.i+self.batch_size)
        idx = self.permutation[self.i:i_max]
        self.i += self.batch_size
        # since the objective is 1/n sum_{i=1}^n (f_i(x) + l2/2*||x||^2)
        # any incomplete minibatch should be normalized by batch_size
        if not self.importance_sampling:
            normalization = self.loss.n / self.steps_per_epoch
        else:
            normalization = self.sample_counts[idx] * self.n_copies / self.steps_per_epoch
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)
        denom_const = 1 / self.lr0
        it_decrease = self.steps_per_epoch * max(0, self.finished_epochs-self.epoch_start_decay)
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*it_decrease**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
        end_of_epoch = self.it%self.steps_per_epoch == self.steps_per_epoch-1
        if end_of_epoch and self.use_prox:
            self.x = self.loss.regularizer.prox(self.x, self.lr * self.steps_per_epoch)
            self.finished_epochs += 1
            
    def should_update_trace(self):
        if not self.update_trace_at_epoch_end:
            super(Shuffling, self).should_update_trace()
        if self.it <= self.save_first_iterations:
            return True
        if self.it%self.steps_per_epoch != 0:
            return False
        self.time_progress = int((self.trace_len-self.save_first_iterations) * self.t / self.t_max)
        self.iterations_progress = int((self.trace_len-self.save_first_iterations) * (self.it / self.it_max))
        enough_progress = max(self.time_progress, self.iterations_progress) > self.max_progress
        return enough_progress
    
    def init_run(self, *args, **kwargs):
        super(Shuffling, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        self.finished_epochs = 0
        self.permutation = np.random.permutation(self.loss.n)
        self.sampled_permutations = 1
