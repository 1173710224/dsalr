from trainers import Trainer
from const import *
from torch.optim import SGD, Adam, Adagrad, RMSprop, Adadelta
from model.blank import BlankLR

if __name__ == "__main__":
    trainer = Trainer()
    optimizer = SGD(trainer.model.parameters(), lr=0.1, weight_decay=1e-4)
    # optimizer = SGD(trainer.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = Adam(trainer.model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = Adagrad(trainer.model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = RMSprop(trainer.model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = Adadelta(trainer.model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = BlankLR(optimizer,)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    pass
