from const import *
from trainers import *
from models import Case_1, Case_2, DeepConv, Fmp, Mlp
from utils import Data
import pickle
data = Data()
loaders = {
    CIFAR10: data.load_cifar10(),
    CIFAR100: data.load_cifar100(),
    MNIST: data.load_mnist(),
    SVHN: data.load_svhn(),
    WINE: data.load_wine(),
    CAR: data.load_car(),
    AGARICUS: data.load_agaricus_lepiota(),
    IRIS: data.load_iris(),
}


class CnnExp():
    def __init__(self) -> None:
        self.datasets = BIG
        self.data = Data()
        pass

    def debug(self, dataset=MNIST, opt=ADAM):
        train_loader, test_loader, input_channel, ndim, nclass = self.data.get(
            dataset)
        model = Fmp(input_channel, ndim, nclass)
        trainer = BatchTrainer(train_loader, test_loader, model)
        if opt == DSA:
            path = f"model/fmp_{dataset}_{opt}"
            trainer.save_model(path)
            trainer.fdsa_train()
        else:
            trainer.train(opt)
        trainer.save_metrics(f"result/big/fmp_{dataset}_{opt}")
        return

    def run(self):
        for dataset in self.datasets:
            train_loader, test_loader, input_channel, ndim, nclass = self.data.get(
                dataset)
            model = Fmp(input_channel, ndim, nclass)
            trainer = BatchTrainer(train_loader, test_loader, model)
            for opt in [SGD, MOMENTUM, ADAM, ADAMAX]:
                trainer.train(opt)
                trainer.save_metrics(f"result/big/fmp_{dataset}_{opt}")
        return


class MlpExp():
    def __init__(self) -> None:
        self.datasets = SMALL
        self.data = Data()
        pass

    def debug(self, dataset=IRIS, opt=ADAM):
        train_data, test_data, input_channel, ndim, nclass = self.data.get(
            dataset)
        model = Mlp(input_channel, ndim, nclass)
        trainer = Trainer(train_data, test_data, model)
        if opt == DSA:
            trainer.fdsa_train()
        else:
            trainer.train(opt)
        trainer.save_metrics(f"result/small/mlp_{dataset}_{opt}")
        return

    def run(self):
        for dataset in self.datasets:
            train_data, test_data, input_channel, ndim, nclass = self.data.get(
                dataset)
            model = Mlp(input_channel, ndim, nclass)
            trainer = Trainer(train_data, test_data, model)
            for opt in OPTIMIZERS:
                if opt == DSA:
                    continue
                trainer.train(opt)
                trainer.save_metrics(f"result/small/mlp_{dataset}_{opt}")
        return

    def run_1000epochs(self):
        EPOCHSDENSE = 1000
        for dataset in self.datasets:
            train_data, test_data, input_channel, ndim, nclass = self.data.get(
                dataset)
            model = Mlp(input_channel, ndim, nclass)
            trainer = Trainer(train_data, test_data, model)
            for opt in OPTIMIZERS:
                if opt == DSA:
                    continue
                trainer.train(opt)
                trainer.save_metrics(f"result/1000epochs/mlp_{dataset}_{opt}")
        return


class SumExp():
    def __init__(self) -> None:
        self.trainer = SumTrainer()
        pass

    def run(self):
        for num in SUMNUMS:
            self.trainer.reset_data(num)
            for opt in OPTIMIZERS:
                self.trainer.train(opt)
        return


class TrackExp():
    def __init__(self) -> None:
        self.trainer = TrackTrainer()
        pass

    def run(self):
        for opt in OPTIMIZERS:
            self.trainer.train(opt)
        return


if __name__ == "__main__":
    pass
