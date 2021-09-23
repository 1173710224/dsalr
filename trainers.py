import torch
from const import *
import torch.nn.functional as F
from optim import Dsa
import warnings
from sklearn.metrics import precision_recall_fscore_support as metrics
import numpy as np
from models import My_loss
import utils
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, train_data, test_data, model, lr=0.001, opt="adam") -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        self.lr = lr
        self.opt = opt
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        self.acc = []
        self.loss = []
        self.p = []
        self.r = []
        self.f1 = []
        self.alpha = []
        pass

    def train(self):
        optimizer = self.get_opt()
        for i in range(EPOCHSDENSE):
            self.model.train()
            x, y = self.x, self.y
            preds = self.model(x)
            loss = F.cross_entropy(preds, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc, p, r, f1 = self.val()
            print("Epoch~{}->loss:{}\nval:{},".format(i +
                  1, loss.item(), acc), end="")
            self.loss.append(loss.item())
            self.p.append(p)
            self.r.append(r)
            self.f1.append(f1)
            self.acc.append(acc)
        return self.acc, self.p, self.r, self.f1, self.loss

    def get_opt(self):
        return utils.get_opt(self.opt, self.model)

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        Preds = preds.detach().cpu().numpy()
        Preds = [np.argmax(Preds[i]) for i in range(len(preds))]
        Y = y.detach().cpu().numpy()
        p, r, f1, _ = metrics(Preds, Y)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu.cpu().numpy(), p, r, f1


class BatchTrainer():
    def __init__(self, train_loader, test_loader, model, lr=0.001, opt=ADAM) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        self.lr = lr
        self.opt = opt

        self.acc = []
        self.loss = []
        self.p = []
        self.r = []
        self.f1 = []
        self.alpha = []
        pass

    def minibatch_train(self):
        self.optimizier = self.get_opt()
        self.model.train()
        for i in range(EPOCHS):
            loss_sum = 0
            img_num = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                preds = self.model(imgs)
                # print(preds[0])
                loss = F.cross_entropy(preds, label)
                # print(loss.item())
                self.optimizier.zero_grad()
                loss.backward()
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
                img_num += len(imgs)
                # print(img_num)
            avg_loss = loss_sum * 1.0/img_num
            acc, p, r, f1 = self.val()
            print("Epoch~{}->loss:{}\nval:{},".format(i +
                  1, avg_loss, acc), end="")
            self.loss.append(avg_loss)
            self.p.append(p)
            self.r.append(r)
            self.f1.append(f1)
            self.acc.append(acc)
        return self.acc, self.p, self.r, self.f1, self.loss

    def batch_train(self):
        self.optimizier = self.get_opt()
        for i in range(EPOCHS):
            self.model.train()
            self.optimizier.zero_grad()
            loss_sum = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)/NUMBATCH
                loss.backward()
                loss_sum += loss.item()
            self.optimizier.step()
            print("Epoch~{}->{}\nval:{}".format(i +
                  1, loss_sum, self.val()), end=",")
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

    def train(self, lr_init, meta_lr):
        # self.optimizier = Dsa(
        #     self.model.parameters(), lr_init, meta_lr)
        # # self.optimizier = torch.optim.Adam(
        # #     self.model.parameters(), lr=self.lr)
        self.optimizier = self.get_opt()
        self.model.train()
        for i in range(EPOCHS):
            loss_sum = 0
            img_num = 0
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                preds = self.model(imgs)
                # print(preds[0])
                loss = F.cross_entropy(preds, label)
                # print(loss.item())
                self.optimizier.zero_grad()
                loss.backward()
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
                img_num += len(imgs)
                # print(img_num)
            avg_loss = loss_sum * 1.0/img_num
            # print("Epoch~{}->{}".format(i+1, avg_loss))
            print("Epoch~{}->{}\nval:{}".format(i+1,
                  avg_loss, self.val()), end=",")
        return

    def get_opt(self):
        return utils.get_opt(self.opt, self.model)


class Case_1_Trainer():
    def __init__(self, train_data, test_data, model, lr=0.001, opt="adam") -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        self.lr = lr
        self.opt = opt
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        self.loss = []
        pass

    def train(self):
        optimizer = self.get_opt()
        for _ in range(EPOCHSDENSE):
            self.model.train()
            x, y = self.x, self.y
            preds = self.model(x)
            loss = F.cross_entropy(preds, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss.append(loss.item())
        return self.loss

    def get_opt(self):
        return utils.get_opt(self.opt, self.model)


class Casee_2_Trainer():
    def __init__(self, x, model, lr=0.001, opt="adam") -> None:
        self.x = x
        self.model = model
        self.lr = lr
        self.opt = opt
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        self.x1 = []
        self.x2 = []
        pass

    def train(self):
        optimizer = self.get_opt()
        for _ in range(EPOCHSDENSE):
            self.model.train()
            x = self.x
            preds = self.model(x)
            loss = My_loss(preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.x = preds
            Preds = preds.detach().cpu().numpy()
            self.x1.append(Preds[0])
            self.x2.append(Preds[1])
        return self.x1, self.x2

    def get_opt(self):
        return utils.get_opt(self.opt, self.model)


if __name__ == "__main__":
    # data = Data()
    # # train_data, test_data, ndim, nclass = data.load_car()
    # # train_data, test_data, ndim, nclass = data.load_wine()
    # # train_data, test_data, ndim, nclass = data.load_iris()
    # train_data, test_data, ndim, nclass = data.load_agaricus_lepiota()
    # model = Mlp(ndim, nclass)
    # trainer = Trainer(train_data, test_data, model)
    # trainer.train()
    # res = trainer.val()
    # print(res)

    # data = Data()
    # # train_loader, test_loader, input_channel, ndim, nclass = data.load_cifar10()
    # train_loader, test_loader, input_channel, ndim, nclass = data.load_mnist()
    # # model = DeepConv(input_channel, ndim, nclass)
    # for lr_init in range(-14, -7):
    #     for meta_lr in [0.00005, 0.0001, 0.0008, 0.001, 0.005, 0.01]:
    #         st = time()
    #         print(f"lr_init:{lr_init}, meta_lr:{meta_lr}")
    #         model = Fmp(input_channel, ndim, nclass)
    #         trainer = BatchTrainer(train_loader, test_loader, model)
    #         trainer.minibatch_train()
    #         print(f"time long: {time() - st}")

    pass
