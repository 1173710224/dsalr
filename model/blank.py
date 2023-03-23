from torch.optim.lr_scheduler import _LRScheduler
import warnings


class BlankLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(BlankLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [group['lr'] for group in self.optimizer.param_groups]
