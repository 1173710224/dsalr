from math import exp
from const import CONFLICT, EPSILON, LOSSNEWLR, LOSSOLDLR
from torch.optim.optimizer import Optimizer
import torch
import torch.nn.functional as F

# 小数据集，-9, 0.6, 0.3


class FDecreaseDsa(Optimizer):
    def __init__(self, params, lr_init=-15, beta_1=0.6, beta_2=0.3) -> None:
        self.params = list(params)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lr_matrix = []
        for param in self.params:
            self.lr_matrix.append(torch.ones(
                param.size(), device=param.device) * lr_init)
        super(FDecreaseDsa, self).__init__(
            self.params, defaults=dict(lr_init=lr_init, beta_1=beta_1, beta_2=beta_2))
        pass

    def w_step_1(self):
        self.base_params = []
        self.base_grads = []
        for i, param in enumerate(self.params):
            self.base_params.append(param.clone())
            self.base_grads.append(param.grad.clone())
            param.data -= torch.mul(self.d(param.grad),
                                    torch.pow(2, self.lr_matrix[i]))
        return

    def lr_step(self):
        for i in range(len(self.lr_matrix)):
            delta = self.d(self.params[i].grad) * self.d(self.base_grads[i])
            self.lr_matrix[i] += 1/2 * delta * \
                (self.beta_1 + self.beta_2) + 1/2 * (self.beta_2 - self.beta_1)
        return

    def w_step_2(self):
        for i, param in enumerate(self.params):
            param.data = self.base_params[i] - torch.mul(
                self.d(self.base_grads[i]), torch.pow(2, self.lr_matrix[i]))
        return

    def d(self, tensor):
        return tensor * (1/(tensor.abs() + EPSILON))


class HypergraDient(Optimizer):
    def __init__(self, params, lr_init=0.001, meta_lr=0.0001) -> None:
        self.params = list(params)
        self.last_w_grad = []
        self.tmp_w_grad = None
        self.lr = lr_init
        self.lr_grad = None
        self.meta_lr = meta_lr

        for param in self.params:
            self.last_w_grad.append(torch.zeros(
                param.size(), device=param.device))
        super(HypergraDient, self).__init__(
            self.params, defaults=dict(lr=lr_init, meta_lr=meta_lr))
        pass

    def step(self, model=None, imgs=None, label=None, closure=None):
        self._lr_autograd()
        self.conflict_dict = self._detect_conflict(model, imgs, label, self.lr,
                                                   self.lr - self.meta_lr * self.lr_grad)
        self.lr -= self.meta_lr * self.lr_grad
        self.param_groups[0]["lr"] = self.lr
        for i, param in enumerate(self.params):
            param.data -= param.grad * self.lr
        return

    def _lr_autograd(self):
        self.tmp_w_grad = []
        for param in self.params:
            if param.grad != None:
                self.tmp_w_grad.append(param.grad.clone())
            else:
                self.tmp_w_grad.append(torch.zeros(
                    param.size(), device=param.device))
        grad = 0
        for i in range(len(self.last_w_grad)):
            grad += - \
                torch.sum(torch.mul(self.last_w_grad[i], self.tmp_w_grad[i]))
        self.last_w_grad = self.tmp_w_grad
        self.lr_grad = grad
        return

    def _detect_conflict(self, model, imgs, label, last_lr, tmp_lr):
        res_dict = {}
        # calculate loss when using old learning rate
        for i, param in enumerate(model.parameters()):
            param.data -= param.grad * last_lr
        preds = model(imgs)
        loss = F.cross_entropy(preds, label)
        res_dict[LOSSOLDLR] = loss.item()
        for i, param in enumerate(model.parameters()):
            param.data += param.grad * last_lr
        # calculate loss when using new learning rate
        for i, param in enumerate(model.parameters()):
            param.data -= param.grad * tmp_lr
        preds = model(imgs)
        loss = F.cross_entropy(preds, label)
        res_dict[LOSSNEWLR] = loss.item()
        for i, param in enumerate(model.parameters()):
            param.data += param.grad * tmp_lr
        if res_dict[LOSSOLDLR] >= res_dict[LOSSNEWLR]:
            res_dict[CONFLICT] = False
        else:
            res_dict[CONFLICT] = True
        return res_dict


class DiffSelfAdapt(Optimizer):
    """
    hypergradient + parameter_specific + direction
    """

    def __init__(self, params, lr_init=0.001, meta_lr=0.0001) -> None:
        self.params = list(params)
        self.last_w_grad = None
        self.tmp_w_grad = None
        self.lr_matrix = []
        self.lr_grad = None
        self.meta_lr = meta_lr
        for param in self.params:
            self.lr_matrix.append(torch.ones(
                param.size(), device=param.device) * lr_init)
        super(DiffSelfAdapt, self).__init__(
            self.params, defaults=dict(lr=lr_init, meta_lr=meta_lr))
        pass

    def step(self, model=None, imgs=None, label=None, closure=None):
        self._w_step()
        preds = model(imgs)
        loss = F.cross_entropy(preds, label)
        self.zero_grad()
        loss.backward()
        self._lr_w_step()
        # detect conflict
        preds = model(imgs)
        newloss = F.cross_entropy(preds, label)
        self.conflict_dict = {}
        self.conflict_dict[LOSSOLDLR] = loss.item()
        self.conflict_dict[LOSSNEWLR] = newloss.item()
        if self.conflict_dict[LOSSOLDLR] >= self.conflict_dict[LOSSNEWLR]:
            self.conflict_dict[CONFLICT] = False
        else:
            self.conflict_dict[CONFLICT] = True
            # self.meta_lr *= 0.9
            # for i in range(len(self.lr_matrix)):
            #     self.lr_matrix[i] *= 0.95
        # avg_lr
        lr_sum = 0
        lr_num = 0
        for lrs in self.lr_matrix:
            lr_sum += lrs.sum().item()
            lr_num += lrs.numel()
        self.param_groups[0]["lr"] = (
            round(0.1/(1 + exp(-lr_sum/lr_num)), 10), round(self.meta_lr, 7))
        return

    def _w_step(self):
        # collect last_w_grad
        self.last_w_grad = []
        for param in self.params:
            if param.grad != None:
                self.last_w_grad.append(param.grad.clone())
            else:
                self.last_w_grad.append(torch.zeros(
                    param.size(), device=param.device))
        # make update for parameter
        for i, param in enumerate(self.params):
            param.data -= torch.mul(self._w_d(param.grad),
                                    self._step_size(self.lr_matrix[i]))
        return

    def _lr_w_step(self):
        # collect tmp_w_grad
        self.tmp_w_grad = []
        for param in self.params:
            if param.grad != None:
                self.tmp_w_grad.append(param.grad.clone())
            else:
                self.tmp_w_grad.append(torch.zeros(
                    param.size(), device=param.device))
        # rollback parameter
        for i, param in enumerate(self.params):
            param.data += torch.mul(
                self._w_d(self.last_w_grad[i]), self._step_size(self.lr_matrix[i]))
        # update learning rate
        # # single learning rate
        # grad = 0
        # max_grad = 0
        # for i in range(len(self.last_w_grad)):
        #     grad += torch.sum(
        #         torch.mul(self.last_w_grad[i], self.tmp_w_grad[i]))
        #     max_grad = max(max_grad, torch.mul(
        #         self.last_w_grad[i], self.tmp_w_grad[i]).max().abs())
        # print("grad: {}, max grad: {}".format(grad, max_grad))
        # for i in range(len(self.last_w_grad)):
        #     self.lr_matrix[i] += self._d(grad) * self.meta_lr

        # # parameter specific
        # for i in range(len(self.last_w_grad)):
        #     self.lr_matrix[i] = self.lr_matrix[i] + self.meta_lr * \
        #         torch.mul(self.last_w_grad[i], self.tmp_w_grad[i])

        # parameter specific learning rate and direction
        for i in range(len(self.last_w_grad)):
            self.lr_matrix[i] += self.meta_lr * \
                self._d(torch.mul(self.last_w_grad[i], self.tmp_w_grad[i]))

        # update parameter
        for i, param in enumerate(self.params):
            param.data -= torch.mul(
                self._w_d(self.last_w_grad[i]), self._zero_step_size(self.lr_matrix[i]))
        # clean grad
        self.last_w_grad.clear()
        self.tmp_w_grad.clear()
        return

    def _d(self, tensor):
        return tensor * (1/(tensor.abs() + EPSILON))
        # return tensor

    def _w_d(self, tensor):
        # return tensor
        return tensor * (1/(tensor.abs() + EPSILON))

    def _step_size(self, tensor):
        return torch.sigmoid(tensor) * 0.1

    def _zero_step_size(self, tensor):
        tensor = torch.sigmoid(tensor) * 0.1
        return torch.mul(tensor, tensor > 0.00001)


# OBSERVW = 0

# # best lr for mlp is -7, 0.7
# # minibatch best lr for mnist is -10, 0.005
# # minibatch best lr for cifar10 is -13, 0.001
# # why accu descrease in the end?
# # minibatch 每一batch拟合太快
# # config:
# # iris: -7, 0.6
# # car: -7, 0.7
# # wine: -8, 0.5
# # agaricus: -8, 0.7
# # cifar10: -13, 0.001
# # cifar100:
# # mnist: -10, 0.001
# # svhn:


# class FDsa(Optimizer):
#     # 更正更新步骤
#     def __init__(self, params, lr_init=-12, meta_lr=0.1) -> None:
#         self.params = list(params)
#         self.meta_lr = meta_lr
#         self.last_w_grad = []
#         self.tmp_w_grad = None
#         self.lr_matrix = []
#         self.lr_grad = None
#         for param in self.params:
#             self.last_w_grad.append(torch.zeros(
#                 param.size(), device=param.device))
#             self.lr_matrix.append(torch.ones(
#                 param.size(), device=param.device) * lr_init)
#         super(FDsa, self).__init__(
#             self.params, defaults=dict(lr=lr_init, meta_lr=meta_lr))
#         pass

#     def w_step_1(self):
#         self.base_params = []
#         self.base_grads = []
#         for i, param in enumerate(self.params):
#             self.base_params.append(param.clone())
#             self.base_grads.append(param.grad.clone())
#             param.data -= torch.mul(self.d(param.grad),
#                                     torch.pow(2, self.lr_matrix[i]))
#         return

#     def lr_step(self):
#         for i in range(len(self.lr_matrix)):
#             self.lr_matrix[i] += self.meta_lr * \
#                 self.d(self.params[i].grad) * self.d(self.base_grads[i])
#         return

#     def w_step_2(self):
#         for i, param in enumerate(self.params):
#             param.data = self.base_params[i] - torch.mul(
#                 self.d(self.base_grads[i]), torch.pow(2, self.lr_matrix[i]))
#         return

#     def d(self, tensor):
#         return tensor * (1/(tensor.abs() + EPSILON))


# class MiniFDsa(Optimizer):
#     # 加入对历史信息的考量
#     def __init__(self, params, lr_init=-13, meta_lr=0.0001, gamma=0.9) -> None:
#         self.params = list(params)
#         self.meta_lr = meta_lr
#         self.gamma = gamma
#         self.lr_matrix = []
#         self.history_delta = []
#         self.lr_grad = None
#         for param in self.params:
#             self.lr_matrix.append(torch.ones(
#                 param.size(), device=param.device) * lr_init)
#             self.history_delta.append(torch.zeros(
#                 param.size(), device=param.device))
#         super(MiniFDsa, self).__init__(
#             self.params, defaults=dict(lr=lr_init, meta_lr=meta_lr, gamma=gamma))
#         pass

#     def w_step_1(self):
#         self.base_params = []
#         self.base_grads = []
#         for i, param in enumerate(self.params):
#             self.base_params.append(param.clone())
#             self.base_grads.append(param.grad.clone())
#             param.data -= torch.mul(self.d(param.grad),
#                                     torch.pow(2, self.lr_matrix[i]))
#         return

#     def lr_step(self):
#         for i in range(len(self.lr_matrix)):
#             self.lr_matrix[i] += self.meta_lr * \
#                 self.d(self.params[i].grad) * self.d(self.base_grads[i])
#         return

#     def w_step_2(self):
#         for i, param in enumerate(self.params):
#             self.history_delta[i] = self.gamma * self.history_delta[i] + (
#                 1 - self.gamma) * (- torch.mul(self.d(self.base_grads[i]), torch.pow(2, self.lr_matrix[i])))
#             param.data = self.base_params[i] + self.history_delta[i]
#         return

#     def d(self, tensor):
#         return tensor * (1/(tensor.abs() + EPSILON))


# class FDecreaseDsa(Optimizer):
#     # 加入慢增快减
#     def __init__(self, params, lr_init=-12, beta_1=0.6, beta_2=0.3) -> None:
#         self.params = list(params)
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.lr_matrix = []
#         self.lr_grad = None
#         for param in self.params:
#             self.lr_matrix.append(torch.ones(
#                 param.size(), device=param.device) * lr_init)
#         super(FDecreaseDsa, self).__init__(
#             self.params, defaults=dict(lr=lr_init))
#         pass

#     def w_step_1(self):
#         self.base_params = []
#         self.base_grads = []
#         for i, param in enumerate(self.params):
#             self.base_params.append(param.clone())
#             self.base_grads.append(param.grad.clone())
#             param.data -= torch.mul(self.d(param.grad),
#                                     torch.pow(2, self.lr_matrix[i]))
#         return

#     def lr_step(self):
#         for i in range(len(self.lr_matrix)):
#             delta = self.d(self.params[i].grad) * self.d(self.base_grads[i])
#             self.lr_matrix[i] += 1/2 * delta * \
#                 (self.beta_1 + self.beta_2) + 1/2 * (self.beta_2 - self.beta_1)
#         return

#     def w_step_2(self):
#         for i, param in enumerate(self.params):
#             param.data = self.base_params[i] - torch.mul(
#                 self.d(self.base_grads[i]), torch.pow(2, self.lr_matrix[i]))
#         return

#     def d(self, tensor):
#         return tensor * (1/(tensor.abs() + EPSILON))


# class FDecreaseMiniDsa(Optimizer):
#     # 慢增快减+历史信息
#     def __init__(self, params, lr_init=-13, beta_1=0.0001, beta_2=0.00005, gamma=0.9) -> None:
#         self.params = list(params)
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.gamma = gamma
#         self.lr_matrix = []
#         self.history_delta = []
#         self.lr_grad = None
#         for param in self.params:
#             self.lr_matrix.append(torch.ones(
#                 param.size(), device=param.device) * lr_init)
#             self.history_delta.append(torch.zeros(
#                 param.size(), device=param.device))
#         super(FDecreaseMiniDsa, self).__init__(
#             self.params, defaults=dict(lr=lr_init))
#         pass

#     def w_step_1(self):
#         self.base_params = []
#         self.base_grads = []
#         for i, param in enumerate(self.params):
#             self.base_params.append(param.clone())
#             self.base_grads.append(param.grad.clone())
#             param.data -= torch.mul(self.d(param.grad),
#                                     torch.pow(2, self.lr_matrix[i]))
#         return

#     def lr_step(self):
#         for i in range(len(self.lr_matrix)):
#             delta = self.d(self.params[i].grad) * self.d(self.base_grads[i])
#             self.lr_matrix[i] += 1/2 * delta * \
#                 (self.beta_1 + self.beta_2) + 1/2 * (self.beta_2 - self.beta_1)
#         return

#     def w_step_2(self):
#         for i, param in enumerate(self.params):
#             self.history_delta[i] = self.gamma * self.history_delta[i] + (
#                 1 - self.gamma) * (- torch.mul(self.d(self.base_grads[i]), torch.pow(2, self.lr_matrix[i])))
#             param.data = self.base_params[i] + self.history_delta[i]
#         return

#     def d(self, tensor):
#         return tensor * (1/(tensor.abs() + EPSILON))


# class MixDsa(Optimizer):
#     # 慢增快减+历史信息
#     def __init__(self, params, lr_init=-12, beta_1=0.6, beta_2=0.3, gamma=0.9) -> None:
#         self.params = list(params)
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.gamma = gamma
#         self.lr_matrix = []
#         self.history_delta = []
#         self.lr_grad = None
#         for param in self.params:
#             self.lr_matrix.append(torch.ones(
#                 param.size(), device=param.device) * lr_init)
#             self.history_delta.append(torch.zeros(
#                 param.size(), device=param.device))
#         super(MixDsa, self).__init__(
#             self.params, defaults=dict(lr=lr_init))
#         pass

#     def minibatch_w_step(self):
#         for i, param in enumerate(self.params):
#             self.history_delta[i] = self.gamma * self.history_delta[i] + (
#                 1 - self.gamma) * (-torch.mul(self.d(param.grad), torch.pow(2, self.lr_matrix[i])))
#             param.data += self.history_delta[i]
#         return

#     def minibatch_w_step_1(self):
#         self.base_params = []
#         self.base_grads = []
#         for i, param in enumerate(self.params):
#             self.base_params.append(param.clone())
#             self.base_grads.append(param.grad.clone())
#             param.data -= torch.mul(self.d(param.grad),
#                                     torch.pow(2, self.lr_matrix[i]))
#         return

#     def minibatch_lr_step(self):
#         for i in range(len(self.lr_matrix)):
#             delta = self.d(self.params[i].grad) * self.d(self.base_grads[i])
#             self.lr_matrix[i] += 1/2 * delta * \
#                 (self.beta_1 + self.beta_2) + 1/2 * (self.beta_2 - self.beta_1)
#         return

#     def minibatch_w_step_2(self):
#         for i, param in enumerate(self.params):
#             self.history_delta[i] = self.gamma * self.history_delta[i] + (
#                 1 - self.gamma) * (-torch.mul(self.d(self.base_grads[i]), torch.pow(2, self.lr_matrix[i])))
#             param.data += self.history_delta[i]
#         return

#     def batch_w_step_1(self):
#         self.base_params = []
#         self.base_grads = []
#         for i, param in enumerate(self.params):
#             self.base_params.append(param.clone())
#             self.base_grads.append(param.grad.clone())
#             param.data -= torch.mul(self.d(param.grad),
#                                     torch.pow(2, self.lr_matrix[i]))
#         return

#     def batch_lr_step(self):
#         for i in range(len(self.lr_matrix)):
#             delta = self.d(self.params[i].grad) * self.d(self.base_grads[i])
#             self.lr_matrix[i] += 1/2 * delta * \
#                 (self.beta_1 + self.beta_2) + 1/2 * (self.beta_2 - self.beta_1)
#         return

#     def batch_w_step_2(self):
#         for i, param in enumerate(self.params):
#             param.data = self.base_params[i] - torch.mul(
#                 self.d(self.base_grads[i]), torch.pow(2, self.lr_matrix[i]))
#         return

#     def d(self, tensor):
#         return tensor * (1/(tensor.abs() + EPSILON))
