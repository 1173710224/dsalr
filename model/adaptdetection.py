from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn.functional as F
import math


class AdaptDectectionLR(_LRScheduler):
    def __init__(self, optimizer: torch.optim.SGD, meta_lr=1e-4, loss_decay=0.99, last_epoch: int = -1, verbose=False) -> None:
        super(AdaptDectectionLR, self).__init__(optimizer, last_epoch, verbose)
        self.last_lr_grad = [0] * len(self.optimizer.param_groups)
        self.accumulated_param_grad = []
        for param in optimizer.param_groups[0]["params"]:
            self.accumulated_param_grad.append(torch.zeros(
                param.size(), device=param.device))
        self.meta_lr = meta_lr
        self.loss_decay = loss_decay
        return

    def get_lr(self) -> float:
        lrs = []
        for lr_idx, group in enumerate(self.optimizer.param_groups):
            lrs.append(group['lr']+self.meta_lr*self.last_lr_grad[lr_idx])
        return list(lrs)

    def step_buffer(self) -> None:
        for lr_idx, group in enumerate(self.optimizer.param_groups):
            grad = 0
            for i, param in enumerate(group["params"]):
                if param.grad != None:
                    grad += torch.sum(torch.mul(self.accumulated_param_grad[i], param.grad.data))
                    self.accumulated_param_grad[i] += param.grad.data
            self.last_lr_grad[lr_idx] = self.last_epoch/(self.last_epoch+1)*self.last_lr_grad[lr_idx]+1/(self.last_epoch+1)*grad
        return


class AdaptDectectionMomentumLR(_LRScheduler):
    def __init__(self, optimizer: torch.optim.SGD, meta_lr=1e-4, loss_decay=0.99, last_epoch: int = -1, verbose=False) -> None:
        super(AdaptDectectionMomentumLR, self).__init__(optimizer, last_epoch, verbose)
        self.last_lr_grad = [0] * len(self.optimizer.param_groups)
        self.accumulated_momentum_buffer = []
        for param in optimizer.param_groups[0]["params"]:
            self.accumulated_momentum_buffer.append(torch.zeros(
                param.size(), device=param.device))
        self.meta_lr = meta_lr
        self.loss_decay = loss_decay
        return

    def get_lr(self) -> float:
        lrs = []
        for lr_idx, group in enumerate(self.optimizer.param_groups):
            lrs.append(group['lr']+self.meta_lr*self.last_lr_grad[lr_idx])
        return list(lrs)

    def step_buffer(self) -> None:
        for lr_idx, group in enumerate(self.optimizer.param_groups):
            grad = 0
            for i, param in enumerate(group["params"]):
                if param.grad != None:
                    grad += torch.sum(torch.mul(self.accumulated_momentum_buffer[i], param.grad.data))
                    self.accumulated_momentum_buffer[i] += self.optimizer.state[i]["momentum_buffer"].data
            self.last_lr_grad[lr_idx] = self.last_epoch/(self.last_epoch+1)*self.last_lr_grad[lr_idx]+1/(self.last_epoch+1)*grad
        return


class AdaptDectectionAdamLR(_LRScheduler):
    def __init__(self, optimizer: torch.optim.SGD, meta_lr=1e-4, loss_decay=0.99, last_epoch: int = -1, verbose=False) -> None:
        super(AdaptDectectionAdamLR, self).__init__(optimizer, last_epoch, verbose)
        self.last_lr_grad = [0] * len(self.optimizer.param_groups)
        self.accumulated_adam_buffer = []
        for param in optimizer.param_groups[0]["params"]:
            self.accumulated_adam_buffer.append(torch.zeros(
                param.size(), device=param.device))
        self.meta_lr = meta_lr
        self.loss_decay = loss_decay
        return

    def get_lr(self) -> float:
        lrs = []
        for lr_idx, group in enumerate(self.optimizer.param_groups):
            lrs.append(group['lr']+self.meta_lr*self.last_lr_grad[lr_idx])
        return list(lrs)

    def step_buffer(self) -> None:
        for lr_idx, group in enumerate(self.optimizer.param_groups):
            grad = 0
            for i, param in enumerate(group["params"]):
                if param.grad != None:
                    grad += torch.sum(torch.mul(self.accumulated_adam_buffer[i], param.grad.data))
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
                    self.accumulated_adam_buffer[i] += numer/denom
            self.last_lr_grad[lr_idx] = self.last_epoch/(self.last_epoch+1)*self.last_lr_grad[lr_idx]+1/(self.last_epoch+1)*grad
        return
