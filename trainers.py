import pickle
import torch
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from const import *
import torch.nn.functional as F
from optim import FDecreaseDsa
import warnings
from sklearn.metrics import precision_recall_fscore_support as metrics
import numpy as np
from models import Mlp, DeepConv, Fmp, Summor, Tracker
from utils import Data
import utils
from time import time
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, train_data, test_data, model) -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
        self.state_dict = {ACCU: [],
                           RECALL: [],
                           PRECISION: [],
                           F1SCORE: [],
                           LOSS: []}
        pass

    def train(self, opt=ADAM):
        self.reset_metrics()
        self.model.reset_parameters()
        optimizer = utils.get_opt(opt, self.model)
        for i in range(EPOCHSDENSE):
            self.model.train()
            preds = self.model(self.x)
            loss = F.cross_entropy(preds, self.y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accu = self.record_metrics(loss.item())
            print("Epoch~{}->loss:{}\nval:{},".format(i +
                  1, loss.item(), accu), end="")
        return

    def fdsa_train(self):
        self.reset_metrics()
        self.model.reset_parameters()
        optimizer = FDecreaseDsa(self.model.parameters())
        for i in range(EPOCHSDENSE):
            self.model.train()

            preds = self.model(self.x)
            loss = F.cross_entropy(preds, self.y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.w_step_1()

            preds = self.model(self.x)
            loss = F.cross_entropy(preds, self.y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.lr_step()
            optimizer.w_step_2()

            accu = self.record_metrics(loss.item())
            print("Epoch~{}->loss:{}\nval:{},".format(i +
                  1, loss.item(), accu), end="")
        return

    def val(self):
        self.model.eval()
        x, y = self.test_data
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        preds = preds.detach().cpu().numpy()
        preds = preds.argmax(axis=1)
        p, r, f1, _ = metrics(preds, y.cpu().numpy())
        return accu.cpu().numpy(), p, r, f1

    def record_metrics(self, loss):
        accu, precision, recall, f1_score = self.val()
        self.state_dict[ACCU].append(accu)
        self.state_dict[PRECISION].append(precision)
        self.state_dict[RECALL].append(recall)
        self.state_dict[F1SCORE].append(f1_score)
        self.state_dict[LOSS].append(loss)
        return accu

    def reset_metrics(self):
        self.state_dict = {ACCU: [],
                           RECALL: [],
                           PRECISION: [],
                           F1SCORE: [],
                           LOSS: []}
        return

    def save_metrics(self, path=""):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict, f)
        return


class BatchTrainer():
    def __init__(self, train_loader, test_loader, model) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()
        self.state_dict = {ACCU: [],
                           RECALL: [],
                           PRECISION: [],
                           F1SCORE: [],
                           LOSS: []}
        self.num_image = 0
        for _, label in self.train_loader:
            self.num_image += len(label)
        pass

    def train(self, opt=ADAM):
        self.reset_metrics()
        self.model.reset_parameters()

        optimizier = utils.get_opt(opt, self.model)
        for i in range(EPOCHS):
            self.model.train()
            loss_sum = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
                loss_sum += loss.item() * len(imgs)/self.num_image
            accu = self.record_metrics(loss_sum)
            print("Epoch~{}->{}\nval:{}".format(i+1,
                  loss_sum, accu), end=",")
        print()
        return

    def fdsa_train(self, path=None, pre_train=True):
        self.reset_metrics()
        self.model.reset_parameters()
        if pre_train:
            # self.optimizier_opt = Adam(self.model.parameters())
            self.optimizier_opt = Adamax(self.model.parameters())
            for i in range(EPOCHSTEP1):
                self.unit_minibatch_train_opt(i)
            print()
            if path != None:
                # self.save_model(path + f"~{self.accu}")
                self.save_model(path)
        else:
            self.load_model(path)
        self.optimizier_opt = None
        self.optimizier_dsa = FDecreaseDsa(self.model.parameters())
        for i in range(EPOCHSTEP2):
            self.unit_batch_train_dsa(i)
        print()
        return

    def unit_minibatch_train_opt(self, epoch=0):
        self.model.train()
        loss_sum = 0
        for imgs, label in self.train_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            preds = self.model(imgs)
            loss = F.cross_entropy(preds, label)
            self.optimizier_opt.zero_grad()
            loss.backward()
            self.optimizier_opt.step()
            loss_sum += loss.item() * len(imgs)/self.num_image
        self.accu = self.record_metrics(loss_sum)
        print("Minibatch~Epoch~{}->loss:{}\nval:{},".format(epoch +
              1, loss_sum, self.accu), end="")
        return

    def unit_batch_train_dsa(self, epoch=0):
        self.optimizier_dsa.zero_grad()
        loss_sum = 0
        for imgs, label in self.train_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            preds = self.model(imgs)
            loss = F.cross_entropy(preds, label) * len(imgs) / self.num_image
            loss.backward()
            loss_sum += loss.item()
        self.optimizier_dsa.w_step_1()

        self.optimizier_dsa.zero_grad()
        loss_sum = 0
        for imgs, label in self.train_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            preds = self.model(imgs)
            loss = F.cross_entropy(preds, label) * len(imgs) / self.num_image
            loss.backward()
            loss_sum += loss.item()
        self.optimizier_dsa.lr_step()
        self.optimizier_dsa.w_step_2()

        accu = self.record_metrics(loss_sum)
        print("Batch~Epoch~{}->loss:{}\nval:{},".format(epoch +
              1, loss_sum, accu), end="")
        return

    def val(self):
        self.model.eval()
        ncorrect = 0
        nsample = 0
        Preds = []
        Y = []
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            preds = self.model(imgs)
            tmp = preds.detach().cpu().numpy()
            Preds.extend([np.argmax(tmp[i]) for i in range(len(tmp))])
            Y.extend(label.detach().cpu().numpy())
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
        p, r, f1, _ = metrics(Preds, Y)
        return (ncorrect/nsample).cpu().numpy(), p, r, f1

    def record_metrics(self, loss_sum):
        accu, precision, recall, f1_score = self.val()
        self.state_dict[ACCU].append(accu)
        self.state_dict[PRECISION].append(precision)
        self.state_dict[RECALL].append(recall)
        self.state_dict[F1SCORE].append(f1_score)
        self.state_dict[LOSS].append(loss_sum)
        return accu

    def save_metrics(self, path=""):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict, f)
        return

    def reset_metrics(self):
        self.state_dict = {ACCU: [],
                           RECALL: [],
                           PRECISION: [],
                           F1SCORE: [],
                           LOSS: []}
        return

    def save_model(self, path="model/tmp"):
        torch.save(self.model.state_dict(), path)
        return

    def load_model(self, path="model/tmp"):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        # self.model = torch.load(path)
        return


class SumTrainer():
    def __init__(self) -> None:
        self.data_num = 10000
        torch.manual_seed(123)
        self.x = torch.rand((self.data_num, 4))
        self.y = torch.sum(self.x, dim=1)
        self.model = Summor()
        if torch.cuda.is_available():
            self.model.cuda()
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        pass

    def train(self, opt=ADAM):
        self.loss = []
        self.params = []
        self.model.train()
        self.model.reset_parameters()
        optimizer = utils.get_opt(opt, self.model)
        for i in range(SUMEPOCH):
            preds = self.model(self.x)
            loss = F.mse_loss(preds.t(), self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss.append(loss.item())
            self.params.append(self.model.dense.weight.data.detach().cpu().numpy())
            print(f"Epoch~{i+1}: {loss.item()}")
        with open(f"result/sum/{self.data_num}_{opt}", "wb") as f:
            pickle.dump({LOSS: self.loss, TRACK: self.params}, f)
        return

    def fdsa_train(self):
        self.loss = []
        self.params = []
        self.model.train()
        self.model.reset_parameters()
        optimizer = utils.get_opt(DSA, self.model)
        for i in range(SUMEPOCH):
            preds = self.model(self.x)
            loss = F.mse_loss(preds.t(), self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.w_step_1()

            preds = self.model(self.x)
            loss = F.mse_loss(preds.t(), self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.lr_step()
            optimizer.w_step_2()

            self.loss.append(loss.item())
            self.params.append(self.model.dense.weight.data.detach().cpu().numpy())
            print(f"Epoch~{i+1}: {loss.item()}")
        with open(f"result/sum/{self.data_num}_dsa", "wb") as f:
            pickle.dump({LOSS: self.loss, TRACK: self.params}, f)
        return

    def reset_data(self, num=10000):
        self.data_num = num
        torch.manual_seed(123)
        self.x = torch.rand((self.data_num, 4))
        self.y = torch.sum(self.x, dim=1)
        if torch.cuda.is_available():
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        return


class TrackTrainer():
    def __init__(self,) -> None:
        self.model = Tracker()
        if torch.cuda.is_available():
            self.model.cuda()
        pass

    def train(self, opt=ADAM):
        self.tracks = []
        self.model.train()
        self.model.reset_parameters()
        optimizer = utils.get_opt(opt, self.model)
        for i in range(TRACKEPOCH):
            preds = self.model()
            optimizer.zero_grad()
            preds.backward()
            optimizer.step()
            self.tracks.append((self.model.w1.detach().cpu().numpy(),
                               self.model.w2.detach().cpu().numpy()))
            print(f"Epoch~{i+1}: {preds.detach().cpu().numpy()}")
        with open(f"result/track/{opt}", "wb") as f:
            pickle.dump(self.tracks, f)
        return

    def fdsa_train(self):
        self.tracks = []
        self.model.train()
        self.model.reset_parameters()
        optimizer = utils.get_opt(DSA, self.model)
        for i in range(TRACKEPOCH):
            preds = self.model()
            optimizer.zero_grad()
            preds.backward()
            optimizer.w_step_1()

            preds = self.model()
            optimizer.zero_grad()
            preds.backward()
            optimizer.lr_step()
            optimizer.w_step_2()

            self.tracks.append((self.model.w1.detach().cpu().numpy(),
                               self.model.w2.detach().cpu().numpy()))
            print(f"Epoch~{i+1}: {preds.detach().cpu().numpy()}")
        with open(f"result/track/dsa", "wb") as f:
            pickle.dump(self.tracks, f)
        return


if __name__ == "__main__":
    print(F.mse_loss(torch.tensor([[1], [2.0]]).float().t(), torch.tensor([1, 2.1]).float()))
    pass
