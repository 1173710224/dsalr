from const import EPSILON
from torch.optim.optimizer import Optimizer
import torch


class FDecreaseDsa(Optimizer):
    def __init__(self, params, lr_init=-12, beta_1=0.6, beta_2=0.3) -> None:
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
