from const import EPSILON
from torch.optim.optimizer import Optimizer
import torch

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


# class Dsa(Optimizer):
#     def __init__(self, params, lr_init=-8, meta_lr=0.5) -> None:
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
#         super(Dsa, self).__init__(
#             self.params, defaults=dict(lr=lr_init, meta_lr=meta_lr))
#         pass

#     def lr_autograd(self):
#         self.lr_grad = []
#         self.tmp_w_grad = []
#         for param in self.params:
#             if param.grad != None:
#                 self.tmp_w_grad.append(param.grad.clone())
#             else:
#                 self.tmp_w_grad.append(torch.zeros(
#                     param.size(), device=param.device))
#         for i in range(len(self.last_w_grad)):
#             grad = -torch.mul(self.last_w_grad[i], self.tmp_w_grad[i])
#             grad = torch.mul(grad, 1/(grad.abs() + EPSILON))
#             self.lr_grad.append(grad)
#         self.last_w_grad = self.tmp_w_grad
#         # print(self.last_w_grad[OBSERVW][0])
#         # print("param", self.params[OBSERVW][0][0])
#         return

#     def step(self, closure=None):
#         self.lr_autograd()
#         # print("alpha grad", self.lr_grad[OBSERVW][0][0])
#         for i in range(len(self.lr_matrix)):
#             self.lr_matrix[i] = self.lr_matrix[i] - \
#                 self.meta_lr * self.lr_grad[i]
#         # print("alpha", self.lr_matrix[OBSERVW][0][0])
#         # print("lr", torch.pow(2, self.lr_matrix[OBSERVW])[0][0])
#         for i, param in enumerate(self.params):
#             param.data -= torch.mul(param.grad * (1/(param.grad.abs() +
#                                     EPSILON)), torch.pow(2, self.lr_matrix[i]))
#         return


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
