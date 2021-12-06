from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Flatten, Linear, PReLU, FractionalMaxPool2d
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from const import *
from utils import Data
import torch


def get_model(model_name, input_channel, ndim, nclass):
    if model_name == FMP:
        model = Fmp(input_channel, ndim, nclass)
    elif model_name == DNN:
        model = DeepConv(input_channel, ndim, nclass)
    elif model_name == RESNET:
        model = ResNet(num_classes=nclass)
    if torch.cuda.is_available():
        model.cuda()
    return model


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
                              out_channels=8, kernel_size=5, bias=False, padding=2)
        self.conv1_2 = Conv2d(in_channels=8,
                              out_channels=8, kernel_size=5, bias=False, padding=2)
        self.conv1_3 = Conv2d(in_channels=8,
                              out_channels=8, kernel_size=2, bias=False, padding=1)
        self.pool1 = AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2_1 = Conv2d(in_channels=8,
                              out_channels=8, kernel_size=2, bias=False, padding=0)
        self.conv2_2 = Conv2d(in_channels=8,
                              out_channels=8, kernel_size=4, bias=False, padding=2)
        self.conv2_3 = Conv2d(in_channels=8,
                              out_channels=8, kernel_size=4, bias=False, padding=1)
        self.pool2 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.conv1_1 = Conv2d(in_channels=input_channel,
        #                       out_channels=4, kernel_size=5, bias=False, padding=2)
        # self.conv1_2 = Conv2d(in_channels=4,
        #                       out_channels=6, kernel_size=5, bias=False, padding=2)
        # self.conv1_3 = Conv2d(in_channels=6,
        #                       out_channels=3, kernel_size=2, bias=False, padding=1)
        # self.pool1 = AvgPool2d(kernel_size=5, stride=1, padding=2)
        # self.conv2_1 = Conv2d(in_channels=3,
        #                       out_channels=6, kernel_size=2, bias=False, padding=0)
        # self.conv2_2 = Conv2d(in_channels=6,
        #                       out_channels=7, kernel_size=4, bias=False, padding=2)
        # self.conv2_3 = Conv2d(in_channels=7,
        #                       out_channels=5, kernel_size=4, bias=False, padding=1)
        # self.pool2 = MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.flatten = Flatten()

        self.dense1 = Linear(
            in_features=8 * self.ndim * self.ndim, out_features=1024*2)
        self.dense3 = Linear(in_features=1024*2, out_features=1024)
        self.dense4 = Linear(in_features=1024, out_features=256)
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
        x = self.dense4(x)
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
        self.dense4.reset_parameters()
        self.dense5.reset_parameters()
        return


class Fmp(nn.Module):
    def __init__(self, input_channel, ndim, nclass):
        super(Fmp, self).__init__()
        self.ndim = ndim
        self.nclass = nclass

        self.relu = nn.LeakyReLU(0.3)
        self.conv1_1 = Conv2d(in_channels=input_channel,
                              out_channels=64, kernel_size=3, bias=False, padding=2)
        self.conv1_2 = Conv2d(in_channels=64,
                              out_channels=64, kernel_size=3, bias=False, padding=2)
        self.pooling1 = FractionalMaxPool2d(kernel_size=3, output_ratio=1/1.6)

        self.conv2_1 = Conv2d(in_channels=64,
                              out_channels=128, kernel_size=3, bias=False, padding=2)
        self.conv2_2 = Conv2d(in_channels=128,
                              out_channels=128, kernel_size=3, bias=False, padding=2)
        self.pooling2 = FractionalMaxPool2d(kernel_size=3, output_ratio=1/1.25)

        self.conv3_1 = Conv2d(in_channels=128,
                              out_channels=256, kernel_size=3, bias=False, padding=2)
        self.conv3_2 = Conv2d(in_channels=256,
                              out_channels=256, kernel_size=3, bias=False, padding=2)
        self.conv3_3 = Conv2d(in_channels=256,
                              out_channels=256, kernel_size=3, bias=False, padding=2)
        self.pooling3 = FractionalMaxPool2d(kernel_size=3, output_ratio=1/1.6)

        self.conv4_1 = Conv2d(in_channels=256,
                              out_channels=256, kernel_size=3, bias=False, padding=2)
        self.conv4_2 = Conv2d(in_channels=256,
                              out_channels=256, kernel_size=3, bias=False, padding=2)
        self.conv4_3 = Conv2d(in_channels=256,
                              out_channels=256, kernel_size=3, bias=False, padding=2)
        self.pooling4 = FractionalMaxPool2d(kernel_size=3, output_ratio=1/1.25)

        self.conv5_1 = Conv2d(in_channels=256,
                              out_channels=512, kernel_size=3, bias=False, padding=2)
        self.conv5_2 = Conv2d(in_channels=512,
                              out_channels=512, kernel_size=3, bias=False, padding=2)
        self.conv5_3 = Conv2d(in_channels=512,
                              out_channels=512, kernel_size=3, bias=False, padding=2)
        self.pooling5 = FractionalMaxPool2d(kernel_size=3, output_ratio=1/1.6)

        self.conv6_1 = Conv2d(in_channels=512,
                              out_channels=512, kernel_size=3, bias=False, padding=2)
        self.conv6_2 = Conv2d(in_channels=512,
                              out_channels=512, kernel_size=3, bias=False, padding=2)
        self.conv6_3 = Conv2d(in_channels=512,
                              out_channels=512, kernel_size=3, bias=False, padding=2)
        self.pooling6 = FractionalMaxPool2d(kernel_size=3, output_ratio=1/1.25)

        self.flatten = nn.Flatten()
        if ndim == 32:
            self.dense1 = Linear(512 * 16 * 16, 1024)
        elif ndim == 28:
            self.dense1 = Linear(512 * 15 * 15, 1024)
        self.dense2 = Linear(1024, 512)
        self.dense3 = Linear(512, nclass)
        return

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.pooling1(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.pooling2(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.pooling3(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.pooling4(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.pooling5(x)

        x = self.conv6_1(x)
        x = self.relu(x)
        x = self.conv6_2(x)
        x = self.relu(x)
        x = self.conv6_3(x)
        x = self.relu(x)
        x = self.pooling6(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x

    def reset_parameters(self):
        self.conv1_1.reset_parameters()

        self.conv1_2.reset_parameters()

        self.conv2_1.reset_parameters()

        self.conv2_2.reset_parameters()

        self.conv3_1.reset_parameters()

        self.conv3_2.reset_parameters()

        self.conv3_3.reset_parameters()

        self.conv4_1.reset_parameters()

        self.conv4_2.reset_parameters()

        self.conv4_3.reset_parameters()

        self.conv5_1.reset_parameters()

        self.conv5_2.reset_parameters()

        self.conv5_3.reset_parameters()

        self.conv6_1.reset_parameters()
        self.conv6_2.reset_parameters()
        self.conv6_3.reset_parameters()

        self.dense1.reset_parameters()
        self.dense2.reset_parameters()
        self.dense3.reset_parameters()
        return


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def reset_parameters(self):
        return


class Summor(nn.Module):
    def __init__(self):
        super(Summor, self).__init__()
        self.dense = Linear(4, 1, bias=False)
        pass

    def reset_parameters(self):
        self.dense.reset_parameters()
        return

    def forward(self, x):
        return self.dense(x)


class Tracker(nn.Module):
    def __init__(self, w1_init=100, w2_init=100):
        super(Tracker, self).__init__()
        self.w1_init = w1_init
        self.w2_init = w2_init
        self.reset_parameters()
        pass

    def forward(self):
        return A*torch.square(self.w1)+B*torch.square(self.w2)

    def reset_parameters(self):
        self.w1 = Parameter(torch.tensor(self.w1_init).float())
        self.w2 = Parameter(torch.tensor(self.w2_init).float())
        return

    def reset_init(self, w1_init, w2_init):
        self.w1_init = w1_init
        self.w2_init = w2_init
        return


if __name__ == "__main__":
    # m = nn.FractionalMaxPool2d(3, output_ratio=0.5)
    # input = torch.randn(20, 16, 50, 32)
    # output = m(input)
    # print(output.size())
    data = Data()
    train_loader, test_loader, input_channel, ndim, nclass = data.load_mnist()
    # train_loader, test_loader, input_channel, ndim, nclass = data.load_cifar10()
    model = Fmp(input_channel, ndim, nclass)
    model.cuda()
    for imgs, label in train_loader:
        model.train()
        imgs = imgs.cuda()
        preds = model(imgs)
        print(preds.size())
    pass
