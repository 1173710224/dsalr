from const import *
from trainers import *
from models import Case_1, Case_2, Fmp, Mlp
from utils import Data
import pickle
data = Data()
loaders = {
    CIFAR10:data.load_cifar10(), 
    CIFAR100:data.load_cifar100(), 
    MNIST:data.load_mnist(),
    SVHN:data.load_svhn(), 
    WINE:data.load_wine(), 
    CAR:data.load_car(),
    AGARICUS:data.load_agaricus_lepiota(), 
    IRIS:data.load_iris(), 
}

def exp1():
    for dataset in BIG:
        train_loader, test_loader, input_channel, ndim, nclass = loaders[dataset]
        BATCHSIZE = batch_size[dataset]
        for op in optimizers:
            if op == DSA:
                continue
            model = Fmp(input_channel, ndim, nclass)
            trainer = BatchTrainer(train_loader, test_loader, model, opt=op)
            res = trainer.minibatch_train()
            # acc, p, r, f1, loss = trainer.minibatch_train()
            with open("result/big/{}-{}".format(op, dataset), "wb") as f:
                pickle.dump(res, f)
            # with open("result/big/{}-{}.txt".format(op, dataset), 'w', encoding='utf8') as f:
            #     f.write("all loss:{}\nall_acc:{}\nall_p:{}\nall_r:{}\nall_f1:{}".format(loss,acc,p,r,f1))

def exp2():
    for dataset in SMALL:
        train_data, test_data, ndim, nclass = loaders[dataset]
        for op in optimizers:
            if op == DSA:
                continue
            best_accu, best_p, best_r, best_f1, best_loss = 0,0,0,0,0
            tot_accu, tot_p, tot_r, tot_f1, tot_loss = [],[],[],[],[]
            model = Mlp(ndim, nclass)
            trainer = Trainer(train_data, test_data, model)
            for i in range(N):
                trainer.model.reset_parameters()
                acc, p, r, f1, loss = trainer.train()
                tot_accu.append(acc)
                tot_p.append(p)
                tot_r.append(r)
                tot_f1.append(f1)
                tot_loss.append(loss)
            loss, acc, p, r, f1 = [],[],[],[],[]
            for j in range(len(tot_p[0])):
                best_accu, best_p, best_r, best_f1, best_loss = 0,0,0,0,0
                for i in range(N):
                     best_accu += tot_accu[i][j] 
                     best_p += tot_p[i][j]
                     best_r += tot_r[i][j]
                     best_f1 += tot_f1[i][j]
                     best_loss += tot_loss[i][j]
                acc.append(best_accu/N)
                loss.append(best_loss/N)
                p.append(best_p/N)
                r.append(best_r/N)
                f1.append(best_f1/N)
            res = {
                "accu":acc,
                "p":p,
                "r":r,
                "f1":f1,
                "loss":loss
            }
            with open("result/small/{}-{}".format(op, dataset), "wb") as f:
                pickle.dump(res, f)
        
def exp2_Many_Epochs():
    for dataset in SMALL:
        train_data, test_data, ndim, nclass = loaders[dataset]
        EPOCHS = MAXEPOCHS
        for op in [ADAM, SGD, MOMENTUM]:
            best_accu, best_p, best_r, best_f1, best_loss = 0,0,0,0,0
            tot_accu, tot_p, tot_r, tot_f1, tot_loss = [],[],[],[],[]
            model = Mlp(ndim, nclass)
            trainer = Trainer(train_data, test_data, model)
            for i in range(N):
                trainer.model.reset_parameters()
                acc, p, r, f1, loss = trainer.train()
                tot_accu.append(acc)
                tot_p.append(p)
                tot_r.append(r)
                tot_f1.append(f1)
                tot_loss.append(loss)
            loss, acc, p, r, f1 = [],[],[],[],[]
            for j in range(len(tot_p[0])):
                best_accu, best_p, best_r, best_f1, best_loss = 0,0,0,0,0
                for i in range(N):
                     best_accu += tot_accu[i][j] 
                     best_p += tot_p[i][j]
                     best_r += tot_r[i][j]
                     best_f1 += tot_f1[i][j]
                     best_loss += tot_loss[i][j]
                acc.append(best_accu/N)
                loss.append(best_loss/N)
                p.append(best_p/N)
                r.append(best_r/N)
                f1.append(best_f1/N)
            res = {
                "accu":acc,
                "p":p,
                "r":r,
                "f1":f1,
                "loss":loss
            }
            with open("result/Epoch-1000/{}-{}".format(op, dataset), "wb") as f:
                pickle.dump(res, f)

def exp3_case1():
    x = torch.rand((10000, 4))
    y = torch.sum(x, dim=1)
    train_data = (x, y)
    test_data = ()
    for op in optimizers:
        best_loss = 0
        tot_loss = []
        model = Case_1()
        trainer = Case_1_Trainer(train_data, test_data, model)
        for i in range(N):
            trainer.model.reset_parameters()
            loss = trainer.train()
            tot_loss.append(loss)
        loss= []
        for j in range(len(tot_loss[0])):
            best_loss = 0
            for i in range(N):
                best_loss += tot_loss[i][j]
            loss.append(best_loss/N)
        res = {"loss":loss}
        with open("result/small/{}-{}".format(op, CASE1), "wb") as f:
            pickle.dump(res, f)

def exp3_case2():
    x = torch.rand((1, 2))
    for op in optimizers:
        model = Case_2()
        trainer = Case_1_Trainer(x, model)
        res = trainer.train()
        with open("result/small/{}-{}".format(op, CASE2), "wb") as f:
            pickle.dump(res, f)
    pass

if __name__ == "__main__":
    exp2()
    exp2_Many_Epochs()
    exp1()
    # exp3_case1()
    # exp3_case2()
