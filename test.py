from model.resnet import ResNet
import torch
from trainers import Trainer
from const import *

# trainer = Trainer(CIFAR100)
model = torch.nn.Linear(10, 10)
# if torch.cuda.is_available():
#     model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
for i in range(10):
    x = torch.rand(10, 10)
    y = model(x).sum()
    optimizer.zero_grad()
    y.backward()
    optimizer.step()
print(optimizer.param_groups)
