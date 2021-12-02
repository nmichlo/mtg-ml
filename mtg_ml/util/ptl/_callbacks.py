import logging
import os.path
import warnings
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from disent.util.visualize.vis_util import make_image_grid
from pytorch_lightning.trainer.supporters import CombinedDataset
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import IterableDataset

from mtg_ml.util.pt import evaluate_context


logger = logging.getLogger(__name__)


# ========================================================================= #
# Vis - Helper                                                              #
# ========================================================================= #


# TODO: move this into disent
@torch.no_grad()
def img_grid_from_batch(batch: torch.Tensor, num_cols: Optional[int] = None, mean_std=None) -> np.ndarray:
    # get num cols
    if num_cols is None:
        num_cols = int(np.ceil(len(batch)**0.5))
    assert num_cols > 0
    # undo normalize
    if mean_std is not None:
        mean, std = mean_std
        batch = (batch * std) + mean  # revert normalize step: xs = (xs - mean) / std
    # convert to uint8
    batch = torch.clip(batch * 255, 0, 255).to(torch.uint8)
    batch = torch.moveaxis(batch, 1, -1)
    batch = batch.detach().cpu().numpy()
    # make grid
    img = make_image_grid(batch, num_cols=num_cols, pad=4)
    # add extra channels & check image size
    if img.shape[-1] == 1:
        img = np.dstack([img, img, img])  # TODO: move this into make_image_grid
    assert img.ndim == 3
    assert img.shape[-1] == 3
    # done!
    return img


# ========================================================================= #
# Visualise                                                                 #
# ========================================================================= #


_DEFAULT = object()


class VisualiseCallbackBase(pl.Callback):
    """
    Takes in an input batch, if the ndim == 4, then it is assumed to be a batch of images (B, C, H, W).
    Feeds the input batch through the model every `period` steps, and obtains the output which now must
    be a batch of images (B, C, H, W).
    """

    def __init__(
        self,
        name: str,
        every_n_steps: int = 1000,
        log_local: bool = True,
        log_wandb: bool = False,
        save_dir: Optional[str] = None,
        mean_std: Optional[Tuple[float, float]] = None,
        figwidth: float = 15,
    ):
        assert log_wandb or log_local or save_dir
        assert isinstance(name, str) and name.strip()
        self._name = name
        self._count = 0  # replace with trainer.global_step?
        self._every_n_steps = every_n_steps
        self._wandb = log_wandb
        self._local = log_local
        self._save_dir = save_dir
        self._figwidth = figwidth
        self._mean_std = mean_std

    @torch.no_grad()
    @rank_zero_only
    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch: torch.Tensor, batch_idx: int, dataloader_idx: int) -> None:
        self._count += 1
        if self._count % self._every_n_steps != 0:
            return
        # skip if nothing should be logged!
        if not (self._wandb or self._local or self._save_dir):
            return
        # produce images
        imgs = self._produce_images(trainer=trainer, pl_module=pl_module)
        # check we produced images
        if not imgs:
            warnings.warn('no images were produced, skipping image visualisations!')
            return
        # save images
        self._save_imgs(imgs, trainer=trainer)

    def _produce_images(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _save_imgs(self, imgs: Dict[str, np.ndarray], trainer: pl.Trainer):
        # check images are RGB
        for k, img in imgs.items():
            assert img.ndim == 3
            assert img.shape[-1] == 3
        # rename images
        imgs = {f'{self._name}_{k}': img for k, img in imgs.items()}

        # plot online
        if self._wandb:
            import wandb
            wandb.log({k: wandb.Image(img) for k, img in imgs.items()})
            logger.info('logged wandb model visualisations')

        # plot local
        if self._local:
            from matplotlib import pyplot as plt
            for k, img in imgs.items():
                w, h = img.shape[:2]
                fig, ax = plt.subplots(figsize=(self._figwidth/w*h, self._figwidth))
                ax.set_title(k)
                ax.imshow(img)
                ax.set_axis_off()
                fig.tight_layout()
                plt.show()
            logger.info('shown matplotlib model visualisations')

        # save locally
        if self._save_dir:
            import imageio
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir, exist_ok=True)
            for k, img in imgs.items():
                path = os.path.abspath(os.path.join(self._save_dir, f'vis_{self._name}_{trainer.global_step}.png'))
                imageio.imsave(path, img)
                logger.info(f'saved model visualisation to: {repr(path)}')

    def _img_grid_from_batch(self, batch: torch.Tensor, num_cols: Optional[int] = None, mean_std=_DEFAULT):
        return img_grid_from_batch(batch, num_cols=num_cols, mean_std=self._mean_std if (mean_std is _DEFAULT) else mean_std)


# ========================================================================= #
# Generic Visualise Callback                                                #
# ========================================================================= #


class VisualiseCallback(VisualiseCallbackBase):

    def __init__(
        self,
        # parent
        name: str,
        # this
        input_batch: Union[torch.Tensor, int, Sequence[int]],
        # parent
        every_n_steps: int = 1000,
        log_local: bool = True,
        log_wandb: bool = False,
        save_dir: Optional[str] = None,
        mean_std: Optional[Tuple[float, float]] = None,
        figwidth: float = 15,
    ):
        super().__init__(name=name, every_n_steps=every_n_steps, log_local=log_local, log_wandb=log_wandb, save_dir=save_dir, mean_std=mean_std, figwidth=figwidth)
        assert isinstance(input_batch, (torch.Tensor, int, Sequence)), f'input_batch should be a torch.Tensor batch or integer representing the number of samples, or a sequence of the indices of the samples themselves, got: {repr(input_batch)}'
        self._input_batch = input_batch

    def _produce_images(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> Dict[str, np.ndarray]:
        imgs = {}
        # get inputs
        inp = self._get_inputs(trainer)
        if inp is None:
            return imgs
        # generate grid
        with evaluate_context(pl_module) as eval_module:
            # move to correct device
            inp = inp.to(eval_module.device)
            # feed forward
            out = eval_module(inp)
            # only create image grid from valid images -- xs can sometimes be representations
            if inp.ndim == 4: imgs['inp'] = self._img_grid_from_batch(inp)
            if out.ndim == 4: imgs['out'] = self._img_grid_from_batch(out)
        # done!
        return imgs

    def _get_inputs(self, trainer: pl.Trainer) -> Optional[torch.Tensor]:
        # get inputs from _input_batch
        # - can be representations (3) or images (4)
        if isinstance(self._input_batch, torch.Tensor):
            xs = self._input_batch
            assert isinstance(xs, torch.Tensor)
            assert xs.ndim in (3, 4)
        # get random images from dataset (4)
        else:  # (int, Sequence):
            dataset = trainer.train_dataloader.dataset
            assert isinstance(dataset, CombinedDataset)
            dataset = dataset.datasets
            if isinstance(dataset, IterableDataset):
                warnings.warn('IterableDataset`s are not supported!')
                return None
            indices = np.random.randint(0, len(dataset), size=self._input_batch) if isinstance(self._input_batch, int) else self._input_batch
            xs = torch.stack([dataset[i] for i in indices], dim=0)
            assert isinstance(xs, torch.Tensor)
            assert xs.ndim == 4
        # done!
        return xs


# ========================================================================= #
# Wandb Setup & Finish Callback                                             #
# ========================================================================= #


class WandbContextManagerCallback(pl.Callback):

    def __init__(self, extra_entries: dict = None):
        self._extra_entries = {} if (extra_entries is None) else extra_entries

    @rank_zero_only
    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        import wandb

        # get initial keys and values
        keys_values = {
            **pl_module.hparams,
        }
        # get batch size from datamodule
        if getattr(getattr(trainer, 'datamodule', None), 'batch_size', None):
            keys_values['batch_size'] = trainer.datamodule.batch_size
        # overwrite keys
        keys_values.update(self._extra_entries)
        wandb.config.update(keys_values, allow_val_change=True)

        print()
        for k, v in keys_values.items():
            print(f'{k}: {repr(v)}')
        print()

    @rank_zero_only
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        import wandb
        wandb.finish()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
