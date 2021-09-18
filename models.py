from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Flatten, Linear, PReLU
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from const import *
from utils import Data
import torch
import math


class BaseMlp(nn.Module):
    def __init__(self, ndim, nclass):
        super(BaseMlp, self).__init__()
        self.ndim = ndim
        self.nclass = nclass

        self.dense1 = Linear(ndim, 64)
        self.dense2 = Linear(64, 256)
        self.dense3 = Linear(256, 128)
        self.dense4 = Linear(128, 32)
        self.dense5 = Linear(32, nclass)
        self.prelu = PReLU()
    pass

    def reset_parameters(self):
        self.dense1.reset_parameters()
        self.dense2.reset_parameters()
        self.dense3.reset_parameters()
        self.dense4.reset_parameters()
        self.dense5.reset_parameters()
        return


class Mlp(BaseMlp):
    def __init__(self, ndim, nclass):
        # super(MLP, self).__init__()
        BaseMlp.__init__(self, ndim, nclass)
    pass

    def forward(self, x):
        x = torch.sigmoid(self.dense1(x))
        x = self.prelu(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))
        x = self.prelu(self.dense4(x))
        x = self.dense5(x)
        return x


class GLMlp(BaseMlp):
    def __init__(self, ndim, nclass):
        BaseMlp.__init__(self, ndim, nclass)
    pass

    def forward(self, x):
        x = torch.sigmoid(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))
        x = torch.sigmoid(self.dense4(x))
        x = self.dense5(x)
        return x


class DeepConv(nn.Module):
    def __init__(self, input_channel, ndim, nclass):
        super(DeepConv, self).__init__()

        self.ndim = ndim
        self.nclass = nclass

        self.conv1_1 = Conv2d(in_channels=input_channel,
                              out_channels=4, kernel_size=5, bias=False, padding=2)
        self.conv1_2 = Conv2d(in_channels=4,
                              out_channels=6, kernel_size=5, bias=False, padding=2)
        self.conv1_3 = Conv2d(in_channels=6,
                              out_channels=3, kernel_size=2, bias=False, padding=1)
        self.pool1 = AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2_1 = Conv2d(in_channels=3,
                              out_channels=6, kernel_size=2, bias=False, padding=0)
        self.conv2_2 = Conv2d(in_channels=6,
                              out_channels=7, kernel_size=4, bias=False, padding=2)
        self.conv2_3 = Conv2d(in_channels=7,
                              out_channels=5, kernel_size=4, bias=False, padding=1)
        self.pool2 = MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.flatten = Flatten()

        self.dense1 = Linear(
            in_features=5 * self.ndim * self.ndim, out_features=1024)
        self.dense3 = Linear(in_features=1024, out_features=256)
        self.dense5 = Linear(in_features=256, out_features=nclass)
        return

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)
        x = self.flatten(x)

        x = F.sigmoid(self.dense1(x))
        x = F.sigmoid(self.dense3(x))
        res = self.dense5(x)
        return res

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv1_3.reset_parameters()
        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        self.conv2_3.reset_parameters()
        self.dense1.reset_parameters()
        self.dense3.reset_parameters()
        self.dense5.reset_parameters()
        return


# todo waiting to be implemented
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        pass


if __name__ == "__main__":
    pass
