import pickle as pkl
import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from const import *
from utils import Data, num_image
from time import time
from model.resnet import ResNet
from torch.optim.lr_scheduler import *
from model.adaptdetection import AdaptDectectionLR, AdaptDectectionAdamLR, AdaptDectectionMomentumLR
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, dataset, device="cuda:0") -> None:
        self.device = device
        # data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # model
        self.model = ResNet(self.input_channel, self.inputdim, self.nclass)
        if torch.cuda.is_available():
            self.model.cuda(self.device)
        self.save_model_path = f"ckpt/resnet_{self.dataset}"
        # optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        pass

    def train(self, load=False, save=False, epochs=EPOCHS):
        if load:
            self.load_model()
        opt_accu = -1
        for i in range(epochs):
            self.model.train()
            loss_sum = 0
            self.scheduler.step()
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                opt_accu = val_accu
                print(
                    f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
            else:
                print(f"Epoch~{i+1}->time:{round(time()-st_time,4)}")
        return

    def val(self):
        self.model.eval()
        ncorrect = 0
        nsample = 0
        valloss = 0
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda(self.device)
                label = label.cuda(self.device)
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
            loss = F.cross_entropy(preds, label)
            valloss += loss.item() * len(imgs)
        valloss = valloss/nsample
        return float((ncorrect/nsample).cpu()), valloss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        return

    def save_model(self, path=None):
        if path:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), self.save_model_path)
        return

    def load_model(self, path=None):
        if path:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(self.save_model_path)
        self.model.load_state_dict(state_dict)
        return


class HDTrainer(Trainer):
    def train(self, load=False, save=False, epochs=EPOCHS):
        if load:
            self.load_model()
        opt_accu = -1
        for i in range(epochs):
            self.model.train()
            loss_sum = 0
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                opt_accu = val_accu
                print(
                    f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
            else:
                print(f"Epoch~{i+1}->time:{round(time()-st_time,4)}")
        return


class ADSTrainer(Trainer):
    def __init__(self, dataset, device="cuda:0") -> None:
        self.device = device
        # data
        self.dataset = dataset
        self.train_loader, self.test_loader, self.input_channel, self.inputdim, self.nclass = Data().get(dataset)
        self.num_image = num_image(self.train_loader)
        # model
        self.model = ResNet(self.input_channel, self.inputdim, self.nclass)
        if torch.cuda.is_available():
            self.model.cuda(self.device)
        self.save_model_path = f"ckpt/resnet_{self.dataset}"
        # optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        pass

    def train(self, load=False, save=False, epochs=EPOCHS):
        if load:
            self.load_model()
        opt_accu = -1
        for i in range(epochs):
            self.model.train()
            loss_sum = 0
            st_time = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda(self.device)
                    label = label.cuda(self.device)
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.buffer()
                loss_sum += loss.item() * len(imgs)/self.num_image
            self.scheduler.step()
            # eval
            val_accu, val_loss = self.val()
            if val_accu > opt_accu:
                opt_accu = val_accu
                print(
                    f"Epoch~{i+1}->train_loss:{round(loss_sum,4)}, val_loss:{round(val_loss, 4)}, val_accu:{round(val_accu, 4)}, time:{round(time()-st_time,4)}")
            else:
                print(f"Epoch~{i+1}->time:{round(time()-st_time,4)}")
        return


if __name__ == "__main__":
    pass
