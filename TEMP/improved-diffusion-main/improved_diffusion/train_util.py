import copy
from typing import Union

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mtg_ml.util.ptl import MlSystem
from .fp16_util import zero_grad
from .gaussian_diffusion import GaussianDiffusion
from .resample import LossAwareSampler, UniformSampler


import pytorch_lightning as pl

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


class TrainLoop:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        diffusion: GaussianDiffusion,
        data: DataLoader,
        batch_size: int,
        microbatch: int,
        lr: float,
        ema_rate: Union[float, str],
        schedule_sampler=None,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = iter(data)

        self.batch_size = batch_size
        self.microbatch = microbatch if (microbatch > 0) else batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0

        self.master_params = list(self.model.parameters())
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]

    def run_loop(self):
        while (not self.lr_anneal_steps) or (self.step < self.lr_anneal_steps):
            batch, cond = next(self.data)
            # start step
            self.forward_backward(batch, cond)
            self.optimize_normal()
            # end step
            self.step += 1

    def forward_backward(self, batch, cond):
        zero_grad(self.master_params)
        # start step

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(device)
            micro_cond = {k: v[i : i + self.microbatch].to(device) for k, v in cond.items()}
            t, weights = self.schedule_sampler.sample(micro.shape[0], device)

            losses = self.diffusion.training_losses(
                model=self.model,
                x_start=micro,
                t=t,
                model_kwargs=micro_cond,
                noise=None,
            )

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()

            # end step
            loss.backward()

    def optimize_normal(self):
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
