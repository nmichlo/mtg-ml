import copy
from typing import Callable
from typing import Dict
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW

from improved_diffusion.lr_schedule import OptimStepSchedule
from improved_diffusion.lr_schedule import OptimStepScheduleLinear
from mtg_ml.util.func import instantiate_required
from mtg_ml.util.ptl import MlSystem
from mtg_ml.util.ptl._callbacks import VisualiseCallbackBase
from .gaussian_diffusion import GaussianDiffusion
from .resample import LossAwareSampler
from .resample import ScheduleSampler
from .script_util import create_diffusion_and_sampler
from .script_util import create_model
from .script_util import DiffusionAndSampleCfg
from .script_util import ImageModelCfg
from .script_util import SrModelCfg


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
        self._lr_scheduler: OptimStepSchedule = None
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

# ========================================================================= #
# Samplers                                                                  #
# ========================================================================= #


@torch.no_grad()
def sample_images(
    system: IDDPM,
    image_size: int = 64,
    use_ddim: bool = False,
    num_samples: int = 16,
    batch_size: int = 16,
    num_classes: Optional[int] = None,
    online: bool = True,
):
    # get the sampling method
    sample_fn = system._diffusion.p_sample_loop if (not use_ddim) else system._diffusion.ddim_sample_loop
    model = system._get_model(online=online)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # sample the required number of values
    all_images, all_labels = [], []
    for _ in range((num_samples + batch_size - 1) // batch_size):
        # generate random classes
        batch_kwargs = {'y': torch.randint(low=0, high=num_classes, size=batch_size, device=system.device)} if (num_classes is not None) else {}
        # sample values
        samples = sample_fn(model, (batch_size, 3, image_size, image_size), clip_denoised=False, progress=True, model_kwargs=batch_kwargs)
        # save values
        all_images.append(samples.detach().cpu())
        if num_classes is not None:
            all_labels.append(batch_kwargs['y'].detach().cpu())
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # concatenate samples
    all_images = torch.cat(all_images, dim=0)[:num_samples]
    if num_classes is None:
        return all_images
    # concatenate labels
    all_labels = torch.cat(all_labels, dim=0)[:num_samples]
    return all_images, all_labels


@torch.no_grad()
def sample_super_res(
    system: IDDPM,
    samples_kwargs: Sequence[dict],
    image_size: int = 256,
    use_ddim: bool = False,
    batch_size: int = 16,
    online: bool = True,
):
    # get the sampling method
    sample_fn = system._diffusion.p_sample_loop if (not use_ddim) else system._diffusion.ddim_sample_loop
    model = system._get_model(online=online)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # sample the required number of values
    all_images = []
    for batch_kwargs in samples_kwargs:
        # get low res image
        batch_kwargs = {'low_res': batch_kwargs['low_res'].to(system.device)}
        # sample values
        samples = sample_fn(model, (batch_size, 3, image_size, image_size), clip_denoised=False, model_kwargs=batch_kwargs)
        # save values
        all_images.append(samples.detach().cpu())
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # concatenate samples
    return torch.cat(all_images, dim=0)


# ========================================================================= #
# Visualise Callback                                                        #
# ========================================================================= #


class IddpmVisualiseCallback(VisualiseCallbackBase):

    def __init__(
        self,
        # parent
        name: str,
        # this
        sample_fn: Callable[[IDDPM, ...], torch.Tensor] = sample_images,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        # parent
        every_n_steps: int = 1000,
        log_local: bool = True,
        log_wandb: bool = False,
        save_dir: Optional[str] = None,
        mean_std: Optional[Tuple[float, float]] = None,
        figwidth: float = 15,
    ):
        super().__init__(name=name, every_n_steps=every_n_steps, log_local=log_local, log_wandb=log_wandb, save_dir=save_dir, mean_std=mean_std, figwidth=figwidth)
        self._sample_fn = sample_fn
        self._sample_kwargs = sample_kwargs or {}

    def _produce_images(self, trainer, pl_module: IDDPM) -> Dict[str, np.ndarray]:
        batch = self._sample_fn(pl_module, **self._sample_kwargs)
        return {'samples': self._img_grid_from_batch(batch)}


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
