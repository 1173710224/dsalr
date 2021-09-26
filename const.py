# 数据集
CIFAR10 = "cifar-10-batches-py"
CIFAR100 = "cifar-100-python"
MNIST = "MNIST"
SVHN = "SVHN"
CAR = "car"
WINE = "wine"
IRIS = "iris"
AGARICUS = "agaricus_lepiota"

DATASETS = [CIFAR10, CIFAR100, MNIST, SVHN, WINE, CAR, IRIS, AGARICUS]
BIG = [MNIST, CIFAR10, CIFAR100, SVHN]
SMALL = [WINE, CAR, IRIS, AGARICUS]

<<<<<<< HEAD
BATCHSIZE = 64
NUMBATCH = 120
EPOCHS = 100
=======

EPOCHS = 30
>>>>>>> 352db27d2decf84d7a9678d1ec6cfd335f87bc78
EPOCHSDENSE = 50
EPOCHSTEP1 = 10
EPOCHSTEP2 = 20
MAXEPOCHS = 1000
SUMEPOCH = 100
TRACKEPOCH = 100

EPSILON = 1e-20
SUMNUMS = [1000, 10000, 1e5]

ADAM = "adam"
DSA = "dsa"
SGD = "sgd"
MOMENTUM = "momentum"
RMSPROP = "rmsprop"
ADAMAX = "adamax"
ADAMW = "adamw"
ADAGRAD = "adagrad"
ADADELTA = "adadelta"
OPTIMIZERS = [
    ADAM, ADAMW, ADAMAX,
    ADADELTA, ADAGRAD, SGD,
    RMSPROP, DSA, MOMENTUM, ]
P_MOMENTUM = 0.9

BATCHSIZE = 64
NAME2BATCHSIZE = {
    CIFAR10: 64,
    CIFAR100: 32,
    MNIST: 64,
    SVHN: 64,
}


NUMIMAGE = {
    MNIST: 60000,
    SVHN: 73257,
    CIFAR10: 50000,
    CIFAR100: 50000
}

ACCU = "accu"
RECALL = "recall"
PRECISION = "precision"
F1SCORE = "f1score"
LOSS = "loss"

if __name__ == "__main__":
    pass
