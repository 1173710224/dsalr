import json
import pickle
import copy
import torch
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from const import *
import torch.nn.functional as F
from optim import FDecreaseDsa
import warnings
from sklearn.metrics import precision_recall_fscore_support as metrics
import numpy as np
from models import Mlp, DeepConv, Fmp, Summor, Tracker, get_model
from utils import Data
import utils
from time import time
warnings.filterwarnings("ignore")


class MiniBatchTrainer():
    """
    specify for a dataset and a model
    """

    def __init__(self, model_name, dataset) -> None:
        # init data
        train_loader, test_loader, input_channel, ndim, nclass = Data().get(dataset)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_image = utils.num_image(train_loader)  # get num of image
        # init model
        self.model = get_model(model_name, input_channel, ndim, nclass)
        # init state dict
        self.state_dict = copy.deepcopy(INITDICT)
        pass

    def train(self, opt):
        self.model.reset_parameters()
        self.optimizier = utils.get_opt(opt, self.model)
        lr_schedular = utils.get_scheduler(opt, self.optimizier)
        epochs = MINIBATCHEPOCHS
        try:
            assert opt == DSA
            epochs = int(epochs/2)
        except:
            pass
        for i in range(epochs):
            self.model.train()
            loss_sum = 0
            begin = time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizier.zero_grad()
                loss.backward()
                # print("base loss:{}".format(round(loss.item(), 5)), end=", ")
                # self.optimizier.step()
                self.optimizier.step(model=self.model, imgs=imgs, label=label)
                self.record_conflict()
                loss_sum += loss.item() * len(imgs)/self.num_image
            self.record_metrics(loss_sum)
            print("Epoch~{}->train_loss:{}, val_loss:{}, val_accu:{}, lr:{}, conflict:{}/{}={}, time:{}s".format(i+1, round(loss_sum, 4),
                  round(self.state_dict[VALLOSS][-1], 4), round(self.state_dict[ACCU][-1], 4), self.optimizier.param_groups[0]['lr'], sum(self.state_dict[CONFLICT]), len(self.state_dict[CONFLICT]), round(sum(self.state_dict[CONFLICT])/(len(self.state_dict[CONFLICT]) + EPSILON), 4), round(time() - begin, 4)))
            try:
                lr_schedular.step()
            except:
                pass
        return

    def val(self):
        self.model.eval()
        ncorrect = 0
        nsample = 0
        valloss = 0
        preds = []
        Y = []
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            tmp_pred = self.model(imgs)
            tmp = tmp_pred.detach().cpu().numpy()
            preds.extend([np.argmax(tmp[i]) for i in range(len(tmp))])
            Y.extend(label.detach().cpu().numpy())
            ncorrect += torch.sum(tmp_pred.max(1)[1].eq(label).double())
            nsample += len(label)
            loss = F.cross_entropy(tmp_pred, label)
            valloss += loss.item() * len(imgs)
        p, r, f1, _ = metrics(preds, Y)
        valloss = valloss/nsample
        return float((ncorrect/nsample).cpu()), p, r, f1, valloss

    def record_metrics(self, loss_sum):
        accu, precision, recall, f1_score, valloss = self.val()
        self.state_dict[ACCU].append(accu)
        self.state_dict[PRECISION].append(list(precision))
        self.state_dict[RECALL].append(list(recall))
        self.state_dict[F1SCORE].append(list(f1_score))
        self.state_dict[VALLOSS].append(valloss)
        self.state_dict[TRAINLOSS].append(loss_sum)
        return

    def record_conflict(self):
        try:
            self.state_dict[LOSSNEWLR].append(
                self.optimizier.conflict_dict[LOSSNEWLR])
            self.state_dict[LOSSOLDLR].append(
                self.optimizier.conflict_dict[LOSSOLDLR])
            self.state_dict[CONFLICT].append(
                self.optimizier.conflict_dict[CONFLICT])
        except:
            pass
        return

    def save_metrics(self, path=""):
        with open(path, "w") as f:
            json.dump(self.state_dict, f)
        return

    def save_model(self, path="model/tmp"):
        torch.save(self.model.state_dict(), path)
        return

    def load_model(self, path="model/tmp"):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        return


class DsaMiniBatchTrainer(MiniBatchTrainer):
    def __init__(self, model_name, dataset) -> None:
        super().__init__(model_name, dataset)

    def train(self):

        return


class Trainer():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        train_data, test_data, ndim, nclass = Data().get(dataset)
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = Mlp(ndim, nclass)
        if torch.cuda.is_available():
            self.model.cuda()
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        self.state_dict = copy.deepcopy(INITDICT)
        return

    def train(self, opt=ADAM):
        self.model.reset_parameters()
        self.optimizer = utils.get_opt(opt, self.model, self.dataset)
        for i in range(EPOCHSDENSE):
            self.model.train()
            preds = self.model(self.x)
            # loss = F.cross_entropy(preds, self.y.long())
            loss = F.mse_loss(torch.softmax(preds, 1),
                              F.one_hot(self.y.long()).float())
            if loss.item() < 0.001:
                break
            self.optimizer.zero_grad()
            loss.backward()
            # self.optimizer.step()
            self.optimizer.step(self.model, self.x, self.y.long())
            self.record_metrics(loss.item())
            print("Epoch~{}->train_loss:{}, val_loss:{}, val_accu:{}, lr:{}, conflict:{}/{}={}".format(i+1, round(loss.item(), 4),
                  round(self.state_dict[VALLOSS][-1], 4), round(self.state_dict[ACCU][-1], 4), self.optimizer.param_groups[0]['lr'], sum(self.state_dict[CONFLICT]), len(self.state_dict[CONFLICT]), round(sum(self.state_dict[CONFLICT])/(len(self.state_dict[CONFLICT]) + EPSILON), 4)))

        return

    def _collect_wrong_cases(self, preds, y):
        flag = preds.max(1)[1].eq(y).double()
        print("wrong cases: ", end=",")
        for index in range(len(flag)):
            if flag[index] == 0:
                print(int(y[index].long()), end=",")
        print()
        return

    def val(self):
        self.model.eval()
        x, y = self.test_data
        preds = self.model(x)
        self._collect_wrong_cases(preds, y)
        # loss = F.cross_entropy(preds, y.long())
        loss = F.mse_loss(torch.softmax(preds, 1), F.one_hot(y.long()).float())
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        preds = preds.detach().cpu().numpy()
        preds = [np.argmax(preds[i]) for i in range(len(preds))]
        p, r, f1, _ = metrics(preds, y.cpu().numpy())
        return float(accu.cpu()), p, r, f1, loss.item()

    def record_metrics(self, loss):
        accu, precision, recall, f1_score, valloss = self.val()
        self.state_dict[ACCU].append(accu)
        self.state_dict[PRECISION].append(list(precision))
        self.state_dict[RECALL].append(list(recall))
        self.state_dict[F1SCORE].append(list(f1_score))
        self.state_dict[VALLOSS].append(valloss)
        self.state_dict[TRAINLOSS].append(loss)
        try:
            self.state_dict[LOSSNEWLR].append(
                self.optimizer.conflict_dict[LOSSNEWLR])
            self.state_dict[LOSSOLDLR].append(
                self.optimizer.conflict_dict[LOSSOLDLR])
            self.state_dict[CONFLICT].append(
                self.optimizer.conflict_dict[CONFLICT])
        except:
            pass
        return accu

    def save_metrics(self, path=""):
        with open(path, "w") as f:
            json.dump(self.state_dict, f)
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
        self.model.train()
        self.model.reset_parameters()
        optimizer = utils.get_opt(opt, self.model)
        for _ in range(SUMEPOCH):
            preds = self.model(self.x)
            loss = F.mse_loss(preds, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss.append(loss.item())
        with open(f"result/sum/{self.data_num}_{opt}", "wb") as f:
            pickle.dump(self.loss, f)
        return

    def reset_data(self, num=10000):
        self.data_num = num
        torch.manual_seed(123)
        self.x = torch.rand((self.data_num, 4))
        self.y = torch.sum(self.x, dim=1)
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
        for _ in range(TRACKEPOCH):
            preds = self.model()
            optimizer.zero_grad()
            preds.backward()
            optimizer.step()
            self.tracks.append((self.model.w1.cpu().numpy(),
                               self.model.w2.cpu().numpy()))
        with open(f"result/track/{opt}", "wb") as f:
            pickle.dump(self.tracks, f)
        return


if __name__ == "__main__":
    # data = Data()
    # # train_data, test_data, ndim, nclass = data.load_car()
    # train_data, test_data, ndim, nclass = data.load_wine()
    # # train_data, test_data, ndim, nclass = data.load_iris()
    # # train_data, test_data, ndim, nclass = data.load_agaricus_lepiota()
    # model = Mlp(ndim, nclass)
    # trainer = Trainer(train_data, test_data, model, opt=ADAM)
    # trainer.fdsa_train()
    # # trainer.train()
    # res = trainer.val()
    # print(res)

    data = Data()
    # train_loader, test_loader, input_channel, ndim, nclass = data.load_cifar10()
    train_loader, test_loader, input_channel, ndim, nclass = data.load_mnist()
    model = Fmp(input_channel, ndim, nclass)
    # model = DeepConv(input_channel, ndim, nclass)
    trainer = MiniBatchTrainer(train_loader, test_loader, model, opt=ADAMAX)
    st = time()
    # trainer.train()
    trainer.load()
    # trainer.fdsa_train()
    # trainer.minibatch_train()
    trainer.fdsa_batch_train(num_image=NUMIMAGE[MNIST])
    # print(trainer.val()[0])
    # trainer.save()
    # trainer.batch_train(num_image=NUMIMAGE[MNIST])
    # trainer.mix_train(num_image=NUMIMAGE[MNIST])
    print(time() - st)
    # for lr_init in range(-14, -7):
    #     for meta_lr in [0.00005, 0.0001, 0.0008, 0.001, 0.005, 0.01]:
    #         st = time()
    #         print(f"lr_init:{lr_init}, meta_lr:{meta_lr}")
    #         model = Fmp(input_channel, ndim, nclass)
    #         trainer = BatchTrainer(train_loader, test_loader, model)
    #         trainer.minibatch_train()
    #         print(f"time long: {time() - st}")
    pass
