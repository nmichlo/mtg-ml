import disent.schedule
import torch


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class OptimStepSchedule(object):

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer

    def _get_lr(self, step: int):
        raise NotImplementedError

    def update(self, step: int):
        lr = self._get_lr(step=step)
        set_optimizer_lr(optimizer=self._optimizer, lr=lr)


class OptimStepScheduleLinear(OptimStepSchedule):

    def __init__(self, optimizer: torch.optim.Optimizer, lr_anneal_steps: int, start_lr: float):
        super().__init__(optimizer=optimizer)
        self._schedule = disent.schedule.LinearSchedule(
            start_step=0,
            end_step=lr_anneal_steps,
            r_start=start_lr,
            r_end=0.0,
        )

    def _get_lr(self, step: int):
        """
        Original Code (has no error handling...):
        >>> frac_done = self.step / self.lr_anneal_steps
        >>> lr = self.lr * (1 - frac_done)
        >>> set_optimizer_lr(self.opt, lr)
        """
        return self._schedule.compute_value(step, 1.0)
