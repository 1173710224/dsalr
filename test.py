from model.resnet import ResNet
import torch
from trainers import Trainer
from const import *

# trainer = Trainer(CIFAR100)
# model = torch.nn.Linear(10, 10)
model = ResNet(3, 32, 100)
# print(type(model.parameters()))
# if torch.cuda.is_available():
#     model.cuda()
param_groups = []
for name, param in model.named_parameters():
    if name.__contains__("bias"):
        param_groups[-1]["params"].append(param)
    else:
        param_groups.append({"params": [param]})
print(param_groups)
# optimizer = torch.optim.Adam(params=param_groups, lr=0.1, weight_decay=1e-4)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
# for i in range(10):
#     x = torch.rand(10, 10)
#     y = model(x).sum()
#     optimizer.zero_grad()
#     y.backward()
#     optimizer.step()
# print(optimizer.param_groups)
