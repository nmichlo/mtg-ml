import disent.schedule
import torch


# ========================================================================= #
# Lr Schedule                                                               #
# ========================================================================= #


class OptimStepScheduleLinear(object):

    def __init__(self, optimizer: torch.optim.Optimizer, lr_anneal_steps: int, start_lr: float):
        self._optimizer = optimizer
        self._schedule = disent.schedule.LinearSchedule(
            start_step=0,
            end_step=lr_anneal_steps,
            r_start=start_lr,
            r_end=0.0,
        )

    def update(self, step: int):
        """
        Original Code (has no error handling...):
        `lr = self.lr * (1 - (self.step / self.lr_anneal_steps))`
        """
        lr = self._schedule.compute_value(step, 1.0)
        # update lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self, step: int):

        return self._schedule.compute_value(step, 1.0)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
