from const import BATCHSIZE, DATASETS
from matplotlib.pyplot import axis
from pandas.core.arrays import sparse
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from torch._C import device, dtype
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt
from bayes_opt import BayesianOptimization
import pandas as pd
import torch


class Data():
    '''
    we load huge datasets as two torch-loader: train-loader and test-loader, so as to train them in multi-batches.
    While load other small or big datasets as
    '''

    def __init__(self) -> None:
        self.datasets = DATASETS
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        return

    def load_cifar10(self):
        data_root_path = "data/"
        train_dataset = datasets.CIFAR10(root=data_root_path, train=True,
                                         transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10(root=data_root_path, train=False,
                                        transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCHSIZE, shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader

    def load_cifar100(self):
        data_root_path = "data/"
        train_dataset = datasets.CIFAR100(root=data_root_path, train=True,
                                          transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR100(root=data_root_path, train=False,
                                         transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCHSIZE, shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader

    def load_mnist(self):
        data_root_path = "data/"
        train_dataset = datasets.MNIST(root=data_root_path, train=True,
                                       transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root=data_root_path, train=False,
                                      transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCHSIZE, shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader

    def load_svhn(self):
        data_root_path = "data/SVHN/"
        train_dataset = datasets.SVHN(root=data_root_path, split="train",
                                      transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.SVHN(root=data_root_path, split="test",
                                     transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCHSIZE, shuffle=True,
                                  num_workers=4)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True)
        return train_loader, test_loader

    def load_wine(self):
        LabelIndex = 0
        path = "data/wine/wine.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, 1:],
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test)

    def load_car(self):
        LabelIndex = 6
        path = "data/car/car.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, :-1]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    pass
