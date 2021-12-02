import copy
from typing import Dict
from typing import Any
from typing import Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW

from lr_schedule import OptimStepSchedule
from lr_schedule import OptimStepScheduleLinear
from mtg_ml.util.func import instantiate_or_none
from mtg_ml.util.func import instantiate_required
from mtg_ml.util.ptl import MlSystem
from .gaussian_diffusion import GaussianDiffusion
from .resample import LossAwareSampler
from .resample import ScheduleSampler
from .resample import UniformSampler


class IDDPM(MlSystem):

    def __init__(
        self, *,
        # targets
            model: Dict[str, Any],
            diffusion: Dict[str, Any],
            schedule_sampler: Optional[Dict[str, Any]] = None,
        # hparams
            lr: float = 1e-4,
            lr_anneal_steps: Optional[int] = None,
            weight_decay: float = 0.0,
            ewm_rate: Optional[float] = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters()
        # placeholder vars
        self._lr_scheduler: OptimStepSchedule = None
        # instantiate
        self._online_model: torch.nn.Module = instantiate_required(self.hparams.model, instance_of=torch.nn.Module)
        self._diffusion: GaussianDiffusion = instantiate_required(self.hparams.diffusion, instance_of=GaussianDiffusion)
        self._sampler: ScheduleSampler = instantiate_or_none(self.hparams.schedule_sampler, instance_of=ScheduleSampler)
        # initialise: exponentially weighted moving average model
        self._target_model: Optional[torch.nn.Module] = None if (self.hparams.ewm_rate is None) else copy.deepcopy(self._online_model)
        # initialise: schedule
        if self._sampler is None:
            self._sampler = UniformSampler(num_timesteps=self._diffusion.num_timesteps)

        #     # instantiate configs
        #     cfg_model = instantiate_required(cfg.system.model_cfg, instance_of=(ImageModelCfg, SrModelCfg))
        #     cfg_diffusion_and_sample = instantiate_required(cfg.system.diffusion_cfg, instance_of=DiffusionAndSampleCfg)
        #     # create model and diffusion
        #     model = create_model(cfg_model)
        #     diffusion, sampler = create_diffusion_and_sampler(cfg_diffusion_and_sample)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # custom scheduler
        if (self.hparams.lr_anneal_steps is not None) and (self.hparams.lr_anneal_steps > 0):
            self._lr_scheduler = OptimStepScheduleLinear(optimizer, lr_anneal_steps=self.hparams.lr_anneal_steps, start_lr=self.hparams.lr)
        # return optimizer, scheduler pair
        return optimizer

    def on_train_batch_start(self, batch: Any, batch_idx: int, **_) -> None:
        # -- this could be moved into a pl.Callback?
        # update the learning rate
        if self._lr_scheduler is not None:
            self._lr_scheduler.update(self.trainer.global_step)

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        batch, cond = batch
        # update scheduler
        t, weights = self._sampler.sample(batch.shape[0], device=self.device)
        # compute loss
        losses = self._diffusion.training_losses(model=self.model, x_start=batch, t=t, model_kwargs=cond, noise=None)
        # update scheduler
        if isinstance(self._sampler, LossAwareSampler):
            self._sampler.update_with_local_losses(t, losses["loss"].detach())
        # compute loss
        return (losses["loss"] * weights).mean()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, **_):
        # -- this could be moved into a pl.Callback?
        # [1. anneal lr, 2. step optimizer, 3. update ewm]
        rate = self.hparams.ewm_rate
        # apply EMA weight update
        for (src_name, src), (trg_name, trg) in zip(self._online_model.named_parameters(), self._target_model.named_parameters()):
            trg.detach().mul_(rate).add_(src, alpha=1-rate)

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
