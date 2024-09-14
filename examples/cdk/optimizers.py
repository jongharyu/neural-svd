# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torchvision
from torch import optim


def get_optimizer_with_params(name, params, lr, momentum, weight_decay):
    if name == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif name == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif name == 'lars':
        optimizer = LARS(
            params=params,
            lr=0,
            weight_decay=weight_decay,  # default weight_decay=1e-6
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm
        )
    else:
        raise NotImplementedError
    return optimizer


def get_optimizer(name, model, lr, momentum, weight_decay):
    return get_optimizer_with_params(name, model.parameters(), lr, momentum, weight_decay)


class LRScheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule, [cosine_lr_schedule[-1]]))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if 'name' in param_group and self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr  # TODO: fix this part
            else:
                param_group['lr'] = self.lr_schedule[self.iter]

        self.current_lr = self.lr_schedule[self.iter]
        self.iter += 1
        return self.current_lr

    def get_lr(self):
        return self.current_lr


if __name__ == '__main__':
    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    epochs = 100
    n_iter = 1000
    scheduler = LRScheduler(optimizer, 10, 1, epochs, 3, 0, n_iter)
    lrs = []
    lr = scheduler.step()


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


class LARS2(optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
            self,
            params,
            lr,
            momentum=0.9,
            use_nesterov=False,
            weight_decay=0.0,
            eta=0.001,
            weight_decay_filter=None,
            lars_adaptation_filter=None
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
            eta=eta,
        )
        super(LARS2, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            lr = group["lr"]
            use_nesterov = group["use_nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                if group['weight_decay_filter'] is None or not group['weight_decay_filter'](param):
                    grad = grad.add(param, alpha=weight_decay)

                trust_ratio = 1.0

                if group['lars_adaptation_filter'] is None or not group['lars_adaptation_filter'](param):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.ge(0),
                        torch.where(
                            g_norm.ge(0),
                            (eta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                scaled_lr = lr * trust_ratio
                if "momentum_buffer" not in param_state:
                    next_v = param_state["momentum_buffer"] = torch.zeros_like(
                        p.data
                    )
                else:
                    next_v = param_state["momentum_buffer"]

                next_v.mul_(momentum).add_(grad, alpha=scaled_lr)
                if use_nesterov:
                    update = (momentum * next_v) + (scaled_lr * grad)
                else:
                    update = next_v
                p.data.add_(-update)
