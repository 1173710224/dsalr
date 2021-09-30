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


EPOCHSDENSE = 100
# fmp mnist:    50  15
# fmp svhn:     75  15
# fmp cifar10:  75  15
# fmp cifar100: 50  15
EPOCHSTEP1 = 100
EPOCHSTEP2 = 10
EPOCHS = EPOCHSTEP1 + EPOCHSTEP2
MAXEPOCHS = 1000
SUMEPOCH = 1000
TRACKEPOCH = 1000

EPSILON = 1e-20
SUMNUMS = [1000, 10000, 100000]

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
    CIFAR10: 128,
    CIFAR100: 256,
    MNIST: 64,
    SVHN: 64,
}

FMP = "fmp"
DNN = "dnn"

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
TRACK = "track"

if __name__ == "__main__":
    pass
