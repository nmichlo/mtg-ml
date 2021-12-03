from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch

from improved_diffusion.framework import IDDPM
from mtg_ml.util.ptl import VisualiseCallbackBase


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
