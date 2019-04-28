import torch
from torch.optim import Optimizer
import numpy as np
from utils.globals import Globals as glob

class Adadelta(Optimizer):
    """Implements Adadelta algorithm.

    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        self.prev_state = dict()
        super(Adadelta, self).__init__(params, defaults)

    # kind of trying to implement restart of [0,1] to avgs. as discussed in:
    # http://akyrillidis.github.io/notes/AdaDelta
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            i = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                # what paper implementation does
                grad = p.grad.data / 2
                if grad.is_sparse:
                    raise RuntimeError('Adadelta does not support sparse gradients')
                state = self.state[p]

                # State initialization
                # force between [0,1]?
                if len(state) == 0:
                    state['step'] = 0
                    # fill with zeros
                    state['square_avg'] = torch.zeros_like(p.data)
                    # initialize to small uniform dist.
                    state['square_avg'] = torch.nn.init.uniform_(state['square_avg'],a=0,b=1)
                    # fill with zeros
                    state['acc_delta'] = torch.zeros_like(p.data)
                    # initialize to small uniform dist.
                    state['acc_delta'] = torch.nn.init.uniform_(state['acc_delta'],a=0,b=1)

                # get current weights
                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho, eps = group['rho'], group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # operations:

                # E[g^2]_t
                square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)
                # sqrt(E[g^2]_t + e)
                std = square_avg.add(eps).sqrt_()
                # dx (here acc_delta = E[dx^2]_{t-1})
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                # update current parameter (x_{t+1} = x_t - lr*dx_t)
                p.data.add_(-group['lr'], delta)
                # acc_delta = E[dx^2]_t
                acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)

                if state['step'] > 1:
                    if torch.dist(self.prev_state[i]['square_avg'], square_avg, 2) < torch.tensor(.01).to(glob.device):
                        state['square_avg'] = torch.nn.init.uniform_(state['square_avg'], a=0, b=1)
                    if torch.dist(self.prev_state[i]['acc_delta'], acc_delta, 2) < torch.tensor(.01).to(glob.device):
                        state['acc_delta'] = torch.nn.init.uniform_(state['acc_delta'], a=0, b=1)

                self.prev_state[i] = dict()
                self.prev_state[i]['square_avg'] = square_avg.clone()
                self.prev_state[i]['acc_delta'] = acc_delta.clone()
                i += 1

        return loss