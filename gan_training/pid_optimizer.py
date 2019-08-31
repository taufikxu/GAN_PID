import torch
from torch.optim.optimizer import Optimizer


class PID_RMSprop(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 vp=0.,
                 vi=0.,
                 vd=0.,
                 centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr,
                        alpha=alpha,
                        eps=eps,
                        centered=centered,
                        vp=vp,
                        vi=vi,
                        vd=vd)
        super(PID_RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PID_RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                alpha = group['alpha']
                vp = group['vp']
                vi = group['vi']
                vd = group['vd']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                    if vi > 0:
                        state['i_buffer'] = torch.zeros_like(p.data)
                    if vd > 0:
                        state['d_buffer'] = p.data

                square_avg = state['square_avg']
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                state['step'] += 1

                if vi > 0:
                    i_buffer = state['i_buffer']
                    i_buffer.add_(p.data)
                else:
                    i_buffer = 0.

                if vd > 0.:
                    d_buffer = state['d_buffer']
                    d_buffer = p.data - d_buffer
                    state['d_buffer'] = p.data
                else:
                    d_buffer = 0.

                controller = vp * p.data + vi * i_buffer + vd * d_buffer
                controller = torch.clamp(controller, -10, 10)
                # grad = grad.add(controller)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(
                        -1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                grad = grad.div(avg)
                grad = grad.add(controller)
                p.data.add_(-group['lr'], grad)

        return loss
