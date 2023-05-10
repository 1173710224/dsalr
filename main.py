from trainers import Trainer, HDTrainer, ADSTrainer
from const import *
from torch.optim import SGD, Adam, Adagrad, RMSprop, Adadelta
from model.blank import BlankLR
from model.hypergradient import HyperGradientLR, HyperGradientAdamLR, HyperGradientMomentumLR
from model.adaptdetection import AdaptDectectionLR, AdaptDectectionAdamLR, AdaptDectectionMomentumLR
import torch

if __name__ == "__main__":
    '''blank'''
    # trainer = Trainer(CIFAR100)
    # optimizer = SGD(trainer.model.parameters(), lr=0.1, weight_decay=1e-4)
    # optimizer = SGD(trainer.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = Adam(trainer.model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = Adagrad(trainer.model.parameters(), lr=0.01, weight_decay=1e-4)
    # optimizer = RMSprop(trainer.model.parameters(), lr=0.01, weight_decay=1e-4)
    # optimizer = Adadelta(trainer.model.parameters(), lr=1, weight_decay=1e-4)
    # scheduler = BlankLR(optimizer,)
    # trainer.set_optimizer(optimizer)
    # trainer.set_scheduler(scheduler)
    # trainer.train()
    # '''manual'''
    # trainer = Trainer(CIFAR100)
    # # optimizer = SGD(trainer.model.parameters(), lr=0.1, weight_decay=1e-4)
    # optimizer = SGD(trainer.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=0.001)
    # trainer.set_optimizer(optimizer)
    # trainer.set_scheduler(scheduler)
    # trainer.train()
    # '''hd'''
    # trainer = HDTrainer(CIFAR100)
    # # optimizer = SGD(trainer.model.parameters(), lr=0.1, weight_decay=1e-4)
    # # optimizer = SGD(trainer.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = Adam(trainer.model.parameters(), lr=0.001, weight_decay=1e-4)
    # # scheduler = HyperGradientLR(optimizer,)
    # # scheduler = HyperGradientMomentumLR(optimizer,)
    # scheduler = HyperGradientAdamLR(optimizer,)
    # trainer.set_optimizer(optimizer)
    # trainer.set_scheduler(scheduler)
    # trainer.train()
    # '''model wise ads'''
    # trainer = ADSTrainer(CIFAR100)
    # optimizer = SGD(trainer.model.parameters(), lr=0.1, weight_decay=1e-4)
    # optimizer = SGD(trainer.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = Adam(trainer.model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = AdaptDectectionLR(optimizer,)
    # scheduler = AdaptDectectionMomentumLR(optimizer,)
    # scheduler = AdaptDectectionAdamLR(optimizer,)
    # trainer.set_optimizer(optimizer)
    # trainer.set_scheduler(scheduler)
    # trainer.train()
    # '''layer wise ads'''
    trainer = ADSTrainer(CIFAR100)
    optimizer = SGD(trainer.model.parameters_layerwise(), lr=0.1, weight_decay=1e-4)
    # optimizer = SGD(trainer.model.parameters_layerwise(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = Adam(trainer.model.parameters_layerwise(), lr=0.001, weight_decay=1e-4)
    scheduler = AdaptDectectionLR(optimizer,)
    # scheduler = AdaptDectectionMomentumLR(optimizer,)
    # scheduler = AdaptDectectionAdamLR(optimizer,)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    trainer.train()
    pass
