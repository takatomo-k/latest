from torch.optim.lr_scheduler import ReduceLROnPlateau

class ReduceLROnPlateauEv(ReduceLROnPlateau) :

    def step(self, metrics, epoch=None):
        """
        modified step from original PyTorch

        return True if the metrics triggers reduce_lr
        """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return True
        else :
            return False
