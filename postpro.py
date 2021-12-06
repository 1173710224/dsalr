from numpy.core.fromnumeric import mean
from const import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import make_interp_spline


class Parser():
    def __init__(self, model, dataset) -> None:
        self.dataset = dataset
        self.model = model
        path = self.get_default_path()
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        pass

    def read(self):
        return self.data

    def reset(self, model, dataset):
        self.dataset = dataset
        self.model = model
        return

    def get_default_path(self):
        if self.dataset in LARGE:
            son_dir = "big"
            # nn = FMP
            nn = DNN
        elif self.dataset in SMALL:
            son_dir = "epochs1000"
            # son_dir = "small"
            nn = MLP
        path = f"result/{son_dir}/{nn}_{self.dataset}_{self.model}"
        return path

    def get_metrics(self):
        print(
            f"{self.data[ACCU][-1]},{mean(self.data[F1SCORE][-1])},{mean(self.data[RECALL][-1])},{mean(self.data[PRECISION][-1])}")
        return


class LossCollect():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        pass

    def get_loss(self):
        losses = {}
        for opt in OPTIMIZERS:
            if opt == RMSPROP:
                continue
            parser = Parser(opt, self.dataset)
            losses[opt] = parser.data[TRAINLOSS]
            # if opt == DSA:
            #     # losses[opt] = parser.data[LOSS]
            #     for i in range(x_len):
            #         if losses[opt][i] < 0.001:
            #             losses[opt][i] = 0
            x_len = len(losses[opt])
        x = [i + 1 for i in range(x_len)]
        return x, losses


def plt_loss(dataset, log=""):
    x, losses = LossCollect(dataset).get_loss()
    if dataset in LARGE:
        for opt, y in losses.items():
            label = OPTIMIZERS2LABEL[opt]
            plt.plot(x, y, linewidth=1.5, label=label)
    else:
        for opt, y in losses.items():
            label = OPTIMIZERS2LABEL[opt]
            plt.plot(x, y, linewidth=1.5, label=label)
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.tick_params(labelsize=15)
    plt.savefig("loss_fig/{}{}.png".format(dataset, log))
    return


class SumProcessor():
    def __init__(self, epoch=1000) -> None:
        self.epoch = epoch
        pass

    def get_data(self, opt):
        path = f"result/sum/{self.epoch}_{opt}"
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def group_losses(self):
        losses = {}
        x_len = self.epoch
        for opt in OPTIMIZERS:
            data = self.get_data(opt)
            losses[opt] = data[TRAINLOSS]
            print(len(losses[opt]))
            for i in range(x_len):
                if losses[opt][i] < 0.000001:
                    losses[opt][i] = 0
        x = [i + 1 for i in range(x_len)]
        return x, losses

    def plt_loss(self, log=""):
        x, losses = self.group_losses()
        for opt, y in losses.items():
            label = OPTIMIZERS2LABEL[opt]
            plt.plot(x, y, linewidth=1.5, label=label)
        plt.yscale("log")
        plt.legend(fontsize=10)
        plt.tick_params(labelsize=15)
        plt.savefig("loss_fig/{}{}.png".format(SUM, self.epoch))
        return

    def group_track(self, index=0):
        tracks = {}
        x_len = self.epoch
        for opt in OPTIMIZERS:
            data = self.get_data(opt)
            data = np.reshape(data[TRACK], (1000, 4))[:, index]
            tracks[opt] = data
            for i in range(x_len):
                if tracks[opt][i] < 0.001:
                    tracks[opt][i] = 0
            # for i in range(x_len):
            #     if losses[opt][i] < 0.000001:
            #         losses[opt][i] = 0
        x = [i + 1 for i in range(x_len)]
        return x, tracks

    def plt_track(self, index=0, log=""):
        x, tracks = self.group_track(index)
        for opt, y in tracks.items():
            label = OPTIMIZERS2LABEL[opt]
            plt.plot(x, y, linewidth=1.5, label=label)
        plt.xlim(0, 150)
        plt.ylim(0.5, 1.5)

        plt.yscale("log")
        # plt.legend(fontsize=10)
        plt.tick_params(labelsize=15)
        plt.savefig("sum_fig/track{}{}.png".format(index, log))
        return


class TrackProcessor():
    def __init__(self) -> None:
        pass

    def get_data(self, opt):
        path = f"result/track/{opt}"
        with open(path, "rb") as f:
            data = pickle.load(f)
        return np.array(data)

    def group_track(self):
        tracks = {}
        for opt in OPTIMIZERS:
            data = self.get_data(opt)
            tracks[opt] = data
            x_len = len(tracks[opt])
        x = [i + 1 for i in range(x_len)]
        return x, tracks

    def height(self, x, y):
        return A * x ** 2 + B * y ** 2

    def plt_track(self, opt=DSA):
        # x = np.linspace(-0.062, 0.005, 300)
        # y = np.linspace(-0.005, 0.005, 300)
        x = np.linspace(-1.05, 0.05*8, 300)
        y = np.linspace(-1.1, 1.1, 300)
        X, Y = np.meshgrid(x, y)
        cset = plt.contourf(X, Y, self.height(
            X, Y), alpha=0.75, cmap=plt.cm.winter)
        plt.colorbar(cset)

        data = self.get_data(opt)
        idx = 0
        plt.plot(data[idx:, 0], data[idx:, 1], c="r", linewidth=0.3)
        plt.scatter(data[idx:, 0], data[idx:, 1], c="r", s=2)
        plt.savefig("track_fig/{}.png".format(opt))
        return


if __name__ == "__main__":
    sp = SumProcessor()
    sp.plt_track(log="detail")

    # tp = TrackProcessor()
    # # tp.plt_track(SGD)
    # tp.plt_track(MOMENTUM)

    # # tp.plt_track(RMSPROP)
    # # tp.plt_track(ADAGRAD)
    # # tp.plt_track(ADADELTA)
    # # tp.plt_track(ADAM)
    # # tp.plt_track(ADAMAX)
    # # tp.plt_track(ADAMW)
    # # tp.plt_track(DSA)
    pass
