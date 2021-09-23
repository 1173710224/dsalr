from const import *
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch


class Data():
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
        return train_loader, test_loader, 3, 32, 10

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
        return train_loader, test_loader, 3, 32, 100

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
        return train_loader, test_loader, 1, 28, 10

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
        return train_loader, test_loader, 3, 32, 10

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
        return (x_train, y_train), (x_test, y_test), 13, 3

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
        return (x_train, y_train), (x_test, y_test), 21, 4

    def load_iris(self):
        LabelIndex = 4
        path = "data/iris/iris.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, :-1],
                                  sp.LabelEncoder().fit_transform(
            df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test), 4, 3

    def load_agaricus_lepiota(self):
        LabelIndex = 0
        path = "data/agaricus-lepiota/agaricus-lepiota.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, 1:11]),
                                   sp.OneHotEncoder(sparse=False).fit_transform(
                                       df.values[:, 12:]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test), 112, 2

def split_Data(X, Y):
    return train_test_split(
            X, Y, test_size=0.2, random_state=0)

if __name__ == "__main__":
    pass
