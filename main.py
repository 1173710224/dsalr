from const import *
from trainers import *
from models import DeepConv, Fmp, Mlp
from utils import Data
from numpy.core.fromnumeric import mean
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CnnExp():
    def __init__(self, model_name=FMP) -> None:
        self.datasets = BIG
        self.data = Data()
        self.model_name = model_name
        pass

    def debug(self, dataset=MNIST, opt=ADAM, pre_train=True):
        print(f"dataset:{dataset},opt:{opt}")
        train_loader, test_loader, input_channel, ndim, nclass = self.data.get(
            dataset)
        if self.model_name == FMP:
            model = Fmp(input_channel, ndim, nclass)
        elif self.model_name == DNN:
            model = DeepConv(input_channel, ndim, nclass)
        trainer = BatchTrainer(train_loader, test_loader, model)
        if opt == DSA:
            path = f"model/pretrained_{self.model_name}_{dataset}_{opt}"
            trainer.fdsa_train(path, pre_train)
        else:
            trainer.train(opt)
        # trainer.save_metrics(f"result/big/{self.model_name}_{dataset}_{opt}")
        print(trainer.state_dict)
        self.data = trainer.state_dict
        index = np.argmax(self.data[ACCU])
        print(f"{round(self.data[ACCU][index] * 100, 2)}\t\
        {round(mean(self.data[F1SCORE][index]) * 100, 2)}\t\
        {round(mean(self.data[RECALL][index]) * 100, 2)}\t\
        {round(mean(self.data[PRECISION][index]) * 100, 2)}")
        return

    def run(self):
        for dataset in self.datasets:

            train_loader, test_loader, input_channel, ndim, nclass = self.data.get(
                dataset)
            if self.model_name == FMP:
                model = Fmp(input_channel, ndim, nclass)
            elif self.model_name == DNN:
                model = DeepConv(input_channel, ndim, nclass)
            trainer = BatchTrainer(train_loader, test_loader, model)
            for opt in [SGD, MOMENTUM, ADAM, ADAMAX]:
                print(f"dataset:{dataset},opt:{opt}")
                trainer.train(opt)
                trainer.save_metrics(
                    f"result/big/{self.model_name}_{dataset}_{opt}")
        return


class MlpExp():
    def __init__(self) -> None:
        self.datasets = SMALL
        self.data = Data()
        pass

    def debug(self, dataset=IRIS, opt=ADAM):
        train_data, test_data, ndim, nclass = self.data.get(dataset)
        model = Mlp(ndim, nclass)
        trainer = Trainer(train_data, test_data, model)
        if opt == DSA:
            trainer.fdsa_train()
        else:
            trainer.train(opt)
        trainer.save_metrics(f"result/small/mlp_{dataset}_{opt}")
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


class SumExp():
    def __init__(self) -> None:
        self.trainer = SumTrainer()
        pass

    def run(self):
        for num in SUMNUMS:
            self.trainer.reset_data(num)
            for opt in OPTIMIZERS:
                if opt == DSA:
                    continue
                self.trainer.train(opt)
        return

    def debug(self, opt=ADAM):
        if opt == DSA:
            self.trainer.fdsa_train()
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
            self.trainer.train(opt)
        return

    def debug(self, opt=ADAM):
        if opt == DSA:
            self.trainer.fdsa_train()
        else:
            self.trainer.train(opt)
        return


if __name__ == "__main__":
    # track_exp = TrackExp()
    # track_exp.trainer.model.reset_init(1, 1)
    # # track_exp.debug(SGD)
    # # track_exp.debug(DSA)
    # track_exp.run()

    # sum_exp = SumExp()
    # # sum_exp.trainer.reset_data(100000)
    # # sum_exp.debug(DSA)
    # sum_exp.run()
    # # sum_exp.debug(SGD)
    # # sum_exp.debug(ADAM)

    # mlp_exp = MlpExp()
    # # mlp_exp.debug(IRIS, DSA)
    # # mlp_exp.run()
    # mlp_exp.debug_1000epochs(CAR, DSA)
    # # mlp_exp.run_1000epochs()

    cnn_exp = CnnExp(model_name=FMP)
    cnn_exp.debug(MNIST, DSA, pre_train=False)
    # cnn_exp.debug(SVHN, ADADELTA)
    # cnn_exp.debug(CIFAR10, DSA)
    # cnn_exp.debug(CIFAR10, ADAM)
    # cnn_exp.debug(CIFAR10, ADAMAX)
    # cnn_exp.debug(CIFAR100, DSA)
    # cnn_exp.debug(CIFAR100, ADAM)
    # cnn_exp.debug(CIFAR100, ADAMAX)

    # cnn_exp.debug(MNIST, ADAMAX)
    # cnn_exp.debug(SVHN, ADAMAX)
    # cnn_exp.debug(CIFAR10, ADAMAX)
    # cnn_exp.debug(CIFAR100, ADAMAX)
    # cnn_exp.debug(MNIST, SGD)
    # cnn_exp.debug(SVHN, SGD)
    # cnn_exp.debug(CIFAR10, SGD)
    # cnn_exp.debug(CIFAR100, SGD)
    pass
