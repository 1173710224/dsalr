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
SUM = "sum"

# sgd: init(-1,1), ab(1,95)
A = 1
# B = 1000
B = 95

EPOCHS = 100
EPOCHSDENSE = 1000
# fmp svhn: 15 15
# fmp mnist: 15 15
EPOCHSTEP1 = 15
EPOCHSTEP2 = 15
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
    RMSPROP, MOMENTUM, DSA]
OPTIMIZERS2LABEL = {
    ADAM: "ADAM",
    ADAMW: "ADAMW",
    ADAMAX: "ADAMAX",
    ADADELTA: "ADADELTA",
    ADAGRAD: "ADAGRAD",
    SGD: "SGD",
    RMSPROP: "RMSPROP",
    MOMENTUM: "MOMENTUM",
    DSA: "DSA"}

P_MOMENTUM = 0.9

BATCHSIZE = 64
NAME2BATCHSIZE = {
    CIFAR10: 64,
    CIFAR100: 32,
    MNIST: 128,
    SVHN: 64,
}

FMP = "fmp"
DNN = "dnn"
MLP = "mlp"

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
