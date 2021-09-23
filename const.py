# 数据集
CAR = "car"
CIFAR10 = "cifar-10-batches-py"
CIFAR100 = "cifar-100-python"
MNIST = "MNIST"

SVHN = "SVHN"
WINE = "wine"
IRIS = "iris"
AGARICUS = "agaricus_lepiota"

DATASETS = [CIFAR10, CIFAR100, MNIST, SVHN, WINE, CAR, IRIS, AGARICUS]
BIG = [MNIST, CIFAR10, CIFAR100, SVHN]
SMALL = [WINE, CAR, IRIS, AGARICUS]
CNNDATASETS = DATASETS[:4]
MLPDATASETS = DATASETS[4:]

BATCHSIZE = 64
NUMBATCH = 120
EPOCHS = 30
EPOCHSDENSE = 100
METALR = 0.01
EPSILON = 1e-20
ADAM = "adam"
DSA = "dsa"
SGD = "sgd"
MOMENTUM = "momentum"
RMSPROP = "rmsprop"
ADAMAX = "adamax"
ADAMW = "adamw"
ADAGRAD = "adagrad"
ADADELTA = "adadelta"
optimizers = [
    ADAM, ADAMW, ADAMAX,
    ADADELTA, ADAGRAD, SGD,
    RMSPROP, DSA, MOMENTUM,]
P_MOMENTUM = 0.9

batch_size = {
    CIFAR10:64, 
    CIFAR100:32, 
    MNIST:64,
    SVHN:64, 
}
N = 3
MAXEPOCHS = 1000
CASE1 = "case1"
CASE2 = "case2"
if __name__ == "__main__":
    pass
