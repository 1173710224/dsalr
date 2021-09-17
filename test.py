from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
from torchvision import transforms
from utils import Data
if __name__ == "__main__":
    data = Data()
    train_loader, test_loader = data.load_cifar100()
    for batch in train_loader:
        imgs, labels = batch
        print(imgs.size(), labels.size())
        break
    pass
