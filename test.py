# from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
# from torchvision import transforms
# from utils import Data
# if __name__ == "__main__":
#     data = Data()
#     train_loader, test_loader = data.load_cifar100()
#     for batch in train_loader:
#         imgs, labels = batch
#         print(imgs.size(), labels.size())
#         break
#     pass

import numpy as np
from utils import Data
import torch
from const import *
import torch.nn.functional as F
from torch.nn import Linear
import warnings
from random import randint
import ctypes
warnings.filterwarnings("ignore")


class LrTrainer():
    def __init__(self, train_data, test_data, model, lr=0.5) -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        self.lr = lr
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.model.train()
        # optimizer = torch.optim.Adagrad(
        #     self.model.parameters(), lr=0.01)
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01)
        # print("init loss:{}\n".format(F.mse_loss(self.model(self.x), self.y)))
        for i in range(EPOCHSDENSE):
            x, y = self.x, self.y
            preds = self.model(x)
            # loss = F.cross_entropy(preds, y.long())
            loss = F.mse_loss(preds, y)
            optimizer.zero_grad()
            loss.backward()
            # for param in self.model.parameters():
            #     print("{},param_grad:{}".format(
            #         param, param.grad))
            # print("Epoch~{}->loss:{}\n".format(i + 1, loss.item()))
            optimizer.step()
        return

    def lr_train(self):
        self.model.train()
        self.last_param = []
        for param in self.model.parameters():
            self.last_param.append(torch.zeros(
                param.size(), device=self.device))
        for i in range(EPOCHSDENSE * 100):
            x, y = self.x, self.y
            preds = self.model(x)
            # loss = F.cross_entropy(preds, y.long())
            loss = F.mse_loss(preds, y)
            loss.backward()
            self.lr_autograd()
            self.lr_step()
            # for param in self.model.parameters():
            #     print("{},param_grad:{}".format(
            #         param, param.grad))
            print("Epoch~{}->lr:{},lr_grad:{},loss:{}\n".format(i + 1,
                                                                self.lr, self.lr_grad, loss.item()))
            # optimizer = torch.optim.Adam(
            #     self.model.parameters(), lr=self.lr)
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr)
            optimizer.step()
            optimizer.zero_grad()
            # if i % 100 == 0:
            #     print("loss:{}".format(loss.item()))
            #     print("current learning rate:{}, grad={}\n".format(
            #         self.lr, self.lr_grad))
        return

    def lr_autograd(self):
        self.tmp_param = []
        for param in self.model.parameters():
            self.tmp_param.append(param.grad.clone())
        grad_sum = 0
        for i in range(len(self.last_param)):
            # print("last_grad:{},tmp_grad:{},hadamaji:{}".format(
            #     self.last_param[i], self.tmp_param[i], torch.mul(self.last_param[i], self.tmp_param[i])))
            grad_sum += torch.sum(
                torch.mul(self.last_param[i], self.tmp_param[i]))
        self.lr_grad = -float(grad_sum)
        # print("grad:{}\n".format(self.lr_grad))
        self.last_param = self.tmp_param
        return

    def lr_step(self):
        self.lr -= METALR * self.lr_grad
        return

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu


class LrMatrixTrainer():
    def __init__(self, train_data, test_data, model, lr=0.5) -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        self.lr = lr
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def lr_train(self):
        self.model.train()
        self.last_param = []
        self.lr_matrix = []
        for param in self.model.parameters():
            self.last_param.append(torch.zeros(
                param.size(), device=self.device))
            self.lr_matrix.append(torch.ones(
                param.size(), device=self.device)/100)
        # print("init loss:{}\n".format(F.mse_loss(self.model(self.x), self.y)))
        for i in range(EPOCHSDENSE):
            x, y = self.x, self.y
            preds = self.model(x)
            # loss = F.cross_entropy(preds, y.long())
            loss = F.mse_loss(preds, y)
            loss.backward()
            self.lr_autograd()
            self.lr_step()
            self.w_step()
            self.zero_grad()
            # print("Epoch~{}->\nlr_matrix:{},\nlr_grad:{},\nloss:{}\n".format(i + 1,
            #                                                                  self.lr_matrix, self.lr_grad, loss.item()))
            # print("Epoch~{}->loss:{}\nlr={}\n".format(i +
            #       1, loss.item(), self.lr_matrix))
        return

    def lr_autograd(self):
        self.lr_grad = []
        self.tmp_param = []
        for param in self.model.parameters():
            self.tmp_param.append(param.grad.clone())
        for i in range(len(self.last_param)):
            self.lr_grad.append(-torch.mul(
                self.last_param[i], self.tmp_param[i]))
        self.last_param = self.tmp_param
        return

    def lr_step(self):
        for i in range(len(self.lr_matrix)):
            self.lr_matrix[i] = self.lr_matrix[i] - METALR * self.lr_grad[i]
        return

    def w_step(self):
        # for param in self.model.parameters():
        #     print("param[0]:{}".format(param[0]))
        #     break
        i = 0
        for param in self.model.parameters():
            param.data -= torch.mul(param.grad, self.lr_matrix[i])
            i += 1
        # for param in self.model.parameters():
        #     print("param[0]:{}".format(param[0]))
        #     break
        return

    def zero_grad(self):
        self.model.zero_grad()
        return

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu


if __name__ == "__main__":
    # trainer = TrainerDense(IRIS, TESTPARAMDENSE)
    # trainer.lr_train()
    # trainer.train()
    # accu = trainer.val()
    # print(accu)
    # model = Dnn(4, 3, TESTPARAMDENSE)

    # model = Linear(4, 1, bias=False)
    # x = torch.rand((1000, 4))
    # y = torch.sum(x, dim=1)
    # print(y[:5])

    # trainer = LrMatrixTrainer((x, y), (x, y), model)
    # trainer.lr_train()
    # trainer.model.train()
    # print(model(x)[:5])
    # for param in model.parameters():
    #     print(param)
    # print(trainer.model(torch.Tensor([[0.1, 0.124, 0.235, 0.8]])))

    # model.reset_parameters()
    # trainer = LrTrainer((x, y), (x, y), model)
    # trainer.train()
    # trainer.model.train()
    # print(trainer.model(x)[:5])
    # for param in model.parameters():
    #     print(param)
    # print(trainer.model(torch.Tensor([[0.1, 0.124, 0.235, 0.8]])))

    # data = Data()
    # train_data, test_data = data.load_iris()
    # ndim = 4
    # nclass = 3
    # model = Dnn(ndim=ndim, nclass=nclass, h_params=TESTPARAMDENSE)
    # trainer = LrMatrixTrainer(train_data, test_data, model)
    # trainer.lr_train()
    # trainer.train()

    # x = torch.rand((2, 3)) - 0.5
    # print(x)
    # print(x.abs())
    # print(1/x * x.abs())
    # print(torch.pow(10, x))

    # print(pow(2, -9+0.001))
    # print(pow(2, -9))
    # print(pow(2, -9-0.001))
    print(0.1 * torch.sigmoid(torch.Tensor([0])))
    pass
