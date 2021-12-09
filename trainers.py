import json
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
        self.state_dict = INITDICT
        pass

    def train(self, opt):
        self.model.reset_parameters()
        self.optimizier = utils.get_opt(opt, self.model)
        lr_schedular = utils.get_scheduler(opt, self.optimizier)
        for i in range(MINIBATCHEPOCHS):
            self.model.train()
            loss_sum = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizier.zero_grad()
                loss.backward()
                self.optimizier.step(model=self.model, imgs=imgs, label=label)
                self.record_conflict()
                loss_sum += loss.item() * len(imgs)/self.num_image
            self.record_metrics(loss_sum)
            print("Epoch~{}->train_loss:{}, val_loss:{}, val_accu:{}, lr:{}, conflict:{}/{}={}".format(i+1, round(loss_sum, 4),
                  round(self.state_dict[VALLOSS][-1], 4), round(self.state_dict[ACCU][-1], 4), self.optimizier.param_groups[0]['lr'], sum(self.state_dict[CONFLICT]), len(self.state_dict[CONFLICT]), round(sum(self.state_dict[CONFLICT])/len(self.state_dict[CONFLICT]), 4)))
            if lr_schedular != None:
                lr_schedular.step()
        return

    def fdsa_train(self):
        self.reset_metrics()
        self.mode.reset_parameters()

        self.optimizier_opt = Adamax(self.model.parameters())
        for i in range(EPOCHSTEP1):
            self.unit_minibatch_train_opt(i)
        print()
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
        accu = self.record_metrics(loss_sum)
        print("Minibatch~Epoch~{}->loss:{}\nval:{},".format(epoch +
              1, loss_sum, accu), end="")
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

    def reset_metrics(self):
        self.state_dict = {ACCU: [],
                           RECALL: [],
                           PRECISION: [],
                           F1SCORE: [],
                           TRAINLOSS: []}
        return

    def save_model(self, path="model/tmp"):
        torch.save(self.model.state_dict(), path)
        return

    def load_model(self, path="model/tmp"):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        return


class Trainer():
    def __init__(self, dataset) -> None:
        train_data, test_data, ndim, nclass = self.data.get(dataset)
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = Mlp(ndim, nclass)
        if torch.cuda.is_available():
            self.model.cuda()
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        self.state_dict = INITDICT
        return

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

            accu = self.record_metrics()
            print("Epoch~{}->loss:{}\nval:{},".format(i +
                  1, loss.item(), accu), end="")
        return

    def val(self):
        self.model.eval()
        x, y = self.test_data
        preds = self.model(x)
        preds = preds.detach().cpu().numpy()
        preds = [np.argmax(preds[i]) for i in range(len(preds))]
        p, r, f1, _ = metrics(preds, y.cpu().numpy())
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu.cpu().numpy(), p, r, f1

    def record_metrics(self, loss):
        accu, precision, recall, f1_score = self.val()
        self.state_dict[ACCU].append(accu)
        self.state_dict[PRECISION].append(precision)
        self.state_dict[RECALL].append(recall)
        self.state_dict[F1SCORE].append(f1_score)
        self.state_dict[TRAINLOSS].append(loss)
        return accu

    def reset_metrics(self):
        self.state_dict = {ACCU: [],
                           RECALL: [],
                           PRECISION: [],
                           F1SCORE: [],
                           TRAINLOSS: []}
        return

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
