
import copy
from typing import Dict
from typing import Any
from typing import Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW

from improved_diffusion.util.lr_schedule import OptimStepScheduleLinear
from mtg_ml.util.func import instantiate_required
from mtg_ml.util.ptl import MlSystem

from improved_diffusion.util.gaussian_diffusion import GaussianDiffusion
from improved_diffusion.util.resample import LossAwareSampler
from improved_diffusion.util.resample import ScheduleSampler
from improved_diffusion.util.factory import create_diffusion_and_sampler
from improved_diffusion.util.factory import create_model
from improved_diffusion.util.factory import DiffusionAndSampleCfg
from improved_diffusion.util.factory import ImageModelCfg
from improved_diffusion.util.factory import SrModelCfg


# ========================================================================= #
# System                                                                    #
# ========================================================================= #


class IDDPM(MlSystem):

    _online_model: torch.nn.Module
    _diffusion: GaussianDiffusion
    _sampler: ScheduleSampler

    def __init__(
        self, *,
        # targets
            cfg_model: Dict[str, Any],
            cfg_diffusion_and_sample: Dict[str, Any],
        # hparams
            lr: float = 1e-4,
            lr_anneal_steps: Optional[int] = None,
            weight_decay: float = 0.0,
            ewm_rate: Optional[float] = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters()
        # placeholder vars
        self._lr_scheduler: OptimStepScheduleLinear = None
        # instantiate configs
        cfg_model = instantiate_required(cfg_model, instance_of=(ImageModelCfg, SrModelCfg))
        cfg_diffusion_and_sample = instantiate_required(cfg_diffusion_and_sample, instance_of=DiffusionAndSampleCfg)
        # instantiate
        self._online_model = create_model(cfg_model)
        self._diffusion, self._sampler = create_diffusion_and_sampler(cfg_diffusion_and_sample)
        # initialise: exponentially weighted moving average model
        self._target_model: Optional[torch.nn.Module] = None if (self.hparams.ewm_rate is None) else copy.deepcopy(self._online_model)

    def configure_optimizers(self):
        optimizer = AdamW(self._online_model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
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
        # The kwargs dict was used for class labels and super res:
        # "y" mapped to values that are integer tensors of class labels.
        # "low_res" mapped to `F.interpolate(large_batch, small_size, mode="area")`
        if isinstance(batch, torch.Tensor):
            batch_kwargs = {}
        else:
            batch, batch_kwargs = batch
        # update scheduler
        t, weights = self._sampler.sample(batch.shape[0], device=self.device)
        # compute loss
        losses = self._diffusion.training_losses(model=self._online_model, x_start=batch, t=t, model_kwargs=batch_kwargs, noise=None)
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
        if self._target_model is not None:
            for (src_name, src), (trg_name, trg) in zip(self._online_model.named_parameters(), self._target_model.named_parameters()):
                trg.detach().mul_(rate).add_(src, alpha=1-rate)

    def forward(self, x, timesteps, online: bool = False, **kwargs):
        model = self._get_model(online=online)
        return model(x, timesteps, **kwargs)  # kwargs: 'y' or 'low_res'

    def _get_model(self, online: bool) -> torch.nn.Module:
        if (not online) and (self._target_model is not None):
            return self._target_model
        else:
            return self._online_model
