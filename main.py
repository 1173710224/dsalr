from torch.utils.data import dataset
from const import *
from trainers import *
from models import DeepConv, Fmp, Mlp, ResNet
from utils import Data
import pickle


class CnnExp():
    def __init__(self,) -> None:
        self.datasets = LARGE
        pass

    def debug(self, model_name=RESNET, dataset=CIFAR10, opt=ADAM, pre_train=True):
        trainer = MiniBatchTrainer(model_name, dataset)
        trainer.train(opt)
        # trainer.enhance_train(mode="batch", opt=opt)
        # trainer.momentum_dsa_train(opt)
        # trainer.fdecreasedsa_enhance_train()
        # trainer.fdecrease_train()
        trainer.save_metrics(
            f"result/large/{model_name}_{dataset}_{opt}_debug.json")
        return

    def run(self):
        for dataset in self.datasets:
            if self.model_name == DNN and dataset == CIFAR100:
                continue
            train_loader, test_loader, input_channel, ndim, nclass = self.data.get(
                dataset)
            if self.model_name == FMP:
                model = Fmp(input_channel, ndim, nclass)
            elif self.model_name == DNN:
                model = DeepConv(input_channel, ndim, nclass)
            trainer = MiniBatchTrainer(train_loader, test_loader, model)
            for opt in [RMSPROP, ADADELTA, ADAMW, ADAGRAD]:
                # for opt in [SGD, MOMENTUM, ADAM, ADAMAX]:
                trainer.train(opt)
                trainer.save_metrics(
                    f"result/big/{self.model_name}_{dataset}_{opt}")
        return

    def find_best_model(self, model_name=RESNET, dataset=CIFAR10):
        trainer = MiniBatchTrainer(model_name, dataset)
        opt = MOMENTUM
        if model_name in [FMP, DNN]:
            opt = ADAMAX
        trainer.pre_train(opt)
        trainer.save_metrics(
            f"result/large/{model_name}_{dataset}_{opt}_debug.json")
        return


class MlpExp():
    def __init__(self) -> None:
        self.datasets = SMALL
        self.data = Data()
        pass

    def debug(self, dataset=IRIS, opt=ADAM, tag=None):
        trainer = Trainer(dataset)
        trainer.train(opt)
        # trainer.fdecrease_train()
        trainer.save_metrics(
            f"result/small/mlp_{dataset}_{opt}{tag}_debug.json")
        return

    def run(self):
        for dataset in self.datasets:
            train_data, test_data, ndim, nclass = self.data.get(
                dataset)
            model = Mlp(ndim, nclass)
            trainer = Trainer(train_data, test_data, model)
            for opt in OPTIMIZERS:
                if opt == DSA:
                    continue
                trainer.train(opt)
                trainer.save_metrics(f"result/small/mlp_{dataset}_{opt}")
        return

    def run_1000epochs(self):
        for dataset in self.datasets:
            train_data, test_data, ndim, nclass = self.data.get(
                dataset)
            model = Mlp(ndim, nclass)
            trainer = Trainer(train_data, test_data, model)
            for opt in OPTIMIZERS:
                if opt == DSA:
                    continue
                trainer.train(opt)
                trainer.save_metrics(f"result/epochs1000/mlp_{dataset}_{opt}")
        return

    def debug_1000epochs(self, dataset=IRIS, opt=ADAM):
        train_data, test_data, ndim, nclass = self.data.get(dataset)
        model = Mlp(ndim, nclass)
        trainer = Trainer(train_data, test_data, model)
        if opt == DSA:
            trainer.fdsa_train()
        else:
            trainer.train(opt)
        trainer.save_metrics(f"result/epochs1000/mlp_{dataset}_{opt}")
        return


class EnhanceExp():
    def __init__(self) -> None:
        pass

    def enhance(self, model_name=RESNET, dataset=MNIST, opt=ADAM, pre_train=True):
        trainer = ModelEnhanceTrainer(model_name, dataset)
        trainer.train(mode="batch", opt=opt)
        trainer.save_metrics(
            f"result/large/{model_name}_{dataset}_{opt}_enhance.json")
        return


class SumExp():
    def __init__(self) -> None:
        self.trainer = SumTrainer()
        pass

    def run(self):
        # for num in SUMNUMS:
        num = 10000
        self.trainer.reset_data(num)
        for opt in OPTIMIZERS:
            if opt in [DSA, HD]:
                continue
            self.trainer.train(opt)
        return

    def debug(self, opt=ADAM):
        if opt in [DSA, HD]:
            self.trainer.dsa_train(opt)
        else:
            self.trainer.train(opt)
        return


class TrackExp():
    def __init__(self) -> None:
        self.trainer = TrackTrainer()
        pass

    def run(self):
        for opt in OPTIMIZERS:
            if opt == DSA:
                continue
            print(opt)
            self.trainer.train(opt)
        return

    def debug(self, opt=ADAM):
        if opt in [DSA, HD]:
            self.trainer.dsa_train(opt)
        else:
            self.trainer.train(opt)
        return


if __name__ == "__main__":
    # track_exp = TrackExp()
    # track_exp.trainer.model.reset_init(-0.06, 0.001)
    # track_exp.trainer.model.reset_init(-1, 1)
    # track_exp.debug(SGD)
    # track_exp.debug(MOMENTUM)
    # track_exp.debug(ADADELTA)
    # track_exp.debug(ADAGRAD)
    # track_exp.debug(RMSPROP)
    # track_exp.debug(ADAM)
    # track_exp.debug(ADAMW)
    # track_exp.debug(ADAMAX)
    # track_exp.debug(HD)
    # track_exp.debug(DSA)

    # track_exp.debug(RMSPROP)
    # track_exp.debug(ADAGRAD)
    # track_exp.debug(ADADELTA)
    # track_exp.debug(ADAM)
    # track_exp.debug(ADAMAX)
    # track_exp.debug(ADAMW)
    # track_exp.debug(DSA)
    # track_exp.run()

    # sum_exp = SumExp()
    # sum_exp.trainer.reset_data(10000)
    # sum_exp.debug(DSA)
    # sum_exp.run()
    # sum_exp.debug(RMSPROP)
    # sum_exp.debug(ADAGRAD)

    mlp_exp = MlpExp()
    opt = DSA
    # mlp_exp.debug(IRIS, opt)
    mlp_exp.debug(CAR, opt, "dotplus")
    # mlp_exp.debug(CAR, opt)
    # mlp_exp.debug(AGARICUS, opt)
    # # mlp_exp.run()
    # # mlp_exp.debug_1000epochs(WINE, DSA)
    # # mlp_exp.run_1000epochs()

    cnn_exp = CnnExp()
    # cnn_exp.find_best_model(DNN, MNIST)
    # cnn_exp.find_best_model(DNN, SVHN)
    # cnn_exp.find_best_model(RESNET, MNIST)
    # cnn_exp.find_best_model(RESNET, SVHN)
    # cnn_exp.find_best_model(RESNET, CIFAR10)
    # cnn_exp.find_best_model(RESNET, CIFAR100)
    # cnn_exp.find_best_model(FMP, MNIST)
    cnn_exp.find_best_model(FMP, SVHN)
    # print("resnet momentum")
    # cnn_exp.debug(dataset=MNIST, opt=MOMENTUM)
    # print("fmp momentum")
    # cnn_exp.debug(model_name=FMP, dataset=MNIST, opt=MOMENTUM)
    # print("fmp adam")
    # cnn_exp.debug(model_name=FMP, dataset=MNIST, opt=ADAM)
    # print("resnet adam")
    # cnn_exp.debug(dataset=MNIST, opt=ADAM)

<<<<<<< HEAD
    # cnn_exp = CnnExp()
    # cnn_exp.debug(model_name=DNN, dataset=MNIST, opt=HD)
    # cnn_exp.debug(model_name=DNN, dataset=SVHN, opt=HD)
    # cnn_exp.debug(model_name=DNN, dataset=SVHN, opt=ADAM)
    # cnn_exp.debug(model_name=DNN, dataset=SVHN, opt=ADAMAX)
    # cnn_exp.debug(model_name=FMP, dataset=MNIST, opt=HD)
    # cnn_exp.debug(model_name=FMP, dataset=SVHN, opt=HD)
    # cnn_exp.debug(model_name=FMP, dataset=SVHN, opt=ADAM)
=======
    cnn_exp = CnnExp()
    # cnn_exp.debug(model_name=DNN, dataset=MNIST, opt=HD)
    # cnn_exp.debug(model_name=DNN, dataset=SVHN, opt=HD)
    cnn_exp.debug(model_name=FMP, dataset=MNIST, opt=ADAMAX)
    cnn_exp.debug(model_name=DNN, dataset=MNIST, opt=ADAM)
    cnn_exp.debug(model_name=DNN, dataset=MNIST, opt=ADAMAX)
    cnn_exp.debug(model_name=RESNET, dataset=MNIST, opt=RMSPROP)
    cnn_exp.debug(model_name=RESNET, dataset=SVHN, opt=RMSPROP)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR10, opt=RMSPROP)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR100, opt=RMSPROP)
    opt = ADADELTA
    cnn_exp.debug(model_name=RESNET, dataset=MNIST, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=SVHN, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR10, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR100, opt=opt)
    opt = ADAGRAD
    cnn_exp.debug(model_name=RESNET, dataset=MNIST, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=SVHN, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR10, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR100, opt=opt)
    opt = ADAMW
    cnn_exp.debug(model_name=RESNET, dataset=MNIST, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=SVHN, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR10, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR100, opt=opt)
    opt = ADAMAX
    cnn_exp.debug(model_name=RESNET, dataset=MNIST, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=SVHN, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR10, opt=opt)
    cnn_exp.debug(model_name=RESNET, dataset=CIFAR100, opt=opt)

>>>>>>> 0a488f4da8afa1232b8da47bbb429377c1ff8a46
    # cnn_exp.debug(dataset=SVHN, opt=DSA)
    # cnn_exp.debug(dataset=CIFAR10, opt=DSA)
    # cnn_exp.debug(dataset=CIFAR100, opt=DSA)

    # cnn_exp.debug(dataset=SVHN, opt=ADAM)
    # cnn_exp.debug(dataset=CIFAR10, opt=ADAM)
    # cnn_exp.debug(dataset=CIFAR100, opt=ADAM)
    # cnn_exp.run()
    # cnn_exp.debug(MNIST, ADAMAX)
    # cnn_exp.debug(SVHN, DSA)
    # cnn_exp.debug(CIFAR10, DSA, pre_train=False)
    # cnn_exp.debug(CIFAR100, DSA)

    # enhance_exp = EnhanceExp()
    # enhance_exp.enhance(model_name=RESNET, dataset=MNIST, opt=DSA)
    pass
