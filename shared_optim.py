import torch
from torch.optim import Optimizer
import torch.multiprocessing as mp


class SharedRMSprop(Optimizer):
    """
    Minor modifications to the PyTorch RMSprop implementation to facilitate multiprocess training
    
    Source Code: 
    https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(SharedRMSprop, self).__init__(params, defaults)

        # Init the states to be shared
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['square_avg'].share_memory_()

                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['momentum_buffer'].share_memory_()

                if group['centered']:
                    state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grad_avg'].share_memory_()

            group["lr"] = mp.Value('d', group["lr"])

    def __setstate__(self, state):
        super(SharedRMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SharedRMSprop does not support sparse gradients')
                state = self.state[p]

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group['lr'].value)
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'].value)

        return loss