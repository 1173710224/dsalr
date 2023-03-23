from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math


class HyperGradientLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, meta_lr=1e-4, last_epoch: int = -1, verbose=False) -> None:
        super(HyperGradientLR, self).__init__(optimizer, last_epoch, verbose)
        self.last_param_grad = []
        for param in optimizer.param_groups[0]["params"]:
            self.last_param_grad.append(torch.zeros(
                param.size(), device=param.device))
        self.meta_lr = meta_lr
        return

    def get_lr(self) -> float:
        lrs = []
        for group in self.optimizer.param_groups:
            grad = 0
            for i, param in enumerate(group["params"]):
                if param.grad != None:
                    grad += torch.sum(torch.mul(self.last_param_grad[i], param.grad.data))
                    self.last_param_grad[i] = param.grad.data
                else:
                    self.last_param_grad[i] = torch.zeros(
                        param.size(), device=param.device)
            lrs.append(group['lr']+self.meta_lr*grad)
        return list(lrs)


class HyperGradientMomentumLR(_LRScheduler):
    def __init__(self, optimizer: torch.optim.SGD, meta_lr=1e-4, last_epoch: int = -1, verbose=False) -> None:
        super(HyperGradientMomentumLR, self).__init__(optimizer, last_epoch, verbose)
        self.last_momentum_buffer = []
        for param in optimizer.param_groups[0]["params"]:
            self.last_momentum_buffer.append(torch.zeros(
                param.size(), device=param.device))
        self.meta_lr = meta_lr
        return

    def get_lr(self) -> float:
        lrs = []
        for group in self.optimizer.param_groups:
            grad = 0
            for i, param in enumerate(group["params"]):
                if param.grad != None:
                    grad += torch.sum(torch.mul(self.last_momentum_buffer[i], param.grad.data))
                    self.last_momentum_buffer[i] = self.optimizer.state[i]["momentum_buffer"].data
                else:
                    self.last_momentum_buffer[i] = torch.zeros(
                        param.size(), device=param.device)
            lrs.append(group['lr']+self.meta_lr*grad)
        return list(lrs)


class HyperGradientAdamLR(_LRScheduler):
    def __init__(self, optimizer: torch.optim.Adam, meta_lr=1e-4, last_epoch: int = -1, verbose=False) -> None:
        super(HyperGradientAdamLR, self).__init__(optimizer, last_epoch, verbose)
        self.last_adam_buffer = []
        for param in optimizer.param_groups[0]["params"]:
            self.last_adam_buffer.append(torch.zeros(
                param.size(), device=param.device))
        self.meta_lr = meta_lr
        return

    def get_lr(self) -> float:
        lrs = []
        for group in self.optimizer.param_groups:
            grad = 0
            for i, param in enumerate(group["params"]):
                if param.grad != None:
                    grad += torch.sum(torch.mul(self.last_adam_buffer[i], param.grad.data))
                    exp_avg = self.optimizer.state[param]["exp_avg"].data
                    exp_avg_sq = self.optimizer.state[param]["exp_avg_sq"].data
                    step = self.optimizer.state[param]["step"]
                    beta1, beta2 = group['betas']
                    weight_decay = group['weight_decay']
                    eps = group["eps"]

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    if weight_decay != 0:
                        grad = grad.add(param, alpha=weight_decay)

                    numer = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).div(bias_correction1)
                    denom = (exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2).sqrt() / math.sqrt(bias_correction2)).add(eps)
                    self.last_adam_buffer[i] = numer/denom
                else:
                    self.last_adam_buffer[i] = torch.zeros(
                        param.size(), device=param.device)
            lrs.append(group['lr']+self.meta_lr*grad)
        return list(lrs)
