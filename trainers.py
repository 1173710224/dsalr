from utils import Data
import torch
from const import *
import torch.nn.functional as F
from models import Mlp, GLMlp, DeepConv
from optim import Dsa
import warnings
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, train_data, test_data, model, lr=0.001) -> None:
        self.x = train_data[0]
        self.y = train_data[1]
        self.test_data = test_data
        self.model = model
        self.lr = lr
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.model.train()
        # optimizer = self.get_opt(ADAM)
        optimizer = self.get_opt(DSA)
        for i in range(EPOCHSDENSE):
            x, y = self.x, self.y
            preds = self.model(x)
            # loss = F.mse_loss(preds, y)
            loss = F.cross_entropy(preds, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch~{}->loss:{}\n".format(i + 1, loss.item()))
            # for param in self.model.parameters():
            #     print(param[0][0])
            #     print(param.grad[0][0])
            #     break
        return

    def get_opt(self, opt="adam"):
        if opt == ADAM:
            return torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
        if opt == DSA:
            return Dsa(self.model.parameters())
        return None

    def val(self):
        x, y = self.test_data
        self.model.eval()
        preds = self.model(x)
        accu = torch.sum(preds.max(1)[1].eq(y).double())/len(y)
        return accu


class BatchTrainer():
    def __init__(self, train_loader, test_loader, model, lr=0.001) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.lr = lr
        pass

    def train(self):
        self.optimizier = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
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
                loss = F.cross_entropy(preds, label)
                self.optimizier.zero_grad()
                loss.backward()
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
                img_num += len(imgs)
                # print(loss.item())
            avg_loss = loss_sum * 1.0/img_num
            print("Epoch~{}->{}".format(i+1, avg_loss))
        return

    def val(self):
        self.model.eval()
        ncorrect = 0
        nsample = 0
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
        return ncorrect/nsample


if __name__ == "__main__":
    # data = Data()
    # train_data, test_data, ndim, nclass = data.load_car()
    # # train_data, test_data, ndim, nclass = data.load_wine()
    # model = Mlp(ndim, nclass)
    # trainer = Trainer(train_data, test_data, model)
    # trainer.train()
    # res = trainer.val()
    # print(res)

    data = Data()
    train_loader, test_loader, input_channel, ndim, nclass = data.load_mnist()
    model = DeepConv(input_channel, ndim, nclass)
    trainer = BatchTrainer(train_loader, test_loader, model)
    trainer.train()
    trainer.val()

    # for name, param in model.named_parameters():
    #     print(name, param)
    pass
