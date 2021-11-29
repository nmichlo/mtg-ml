import logging
import os.path
import warnings
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
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from mtg_ml.util.pt import evaluate_context


logger = logging.getLogger(__name__)


# ========================================================================= #
# Visualise                                                                 #
# ========================================================================= #


class VisualiseCallback(pl.Callback):
    """
    Takes in an input batch, if the ndim == 4, then it is assumed to be a batch of images (B, C, H, W).
    Feeds the input batch through the model every `period` steps, and obtains the output which now must
    be a batch of images (B, C, H, W).
    """

    def __init__(
        self,
        name: str,
        input_batch: Union[torch.Tensor, int, Sequence[int]],
        every_n_steps: int = 1000,
        log_local: bool = True,
        log_wandb: bool = False,
        save_dir: Optional[str] = None,
        mean_std: Optional[Tuple[float, float]] = None,
        figwidth: float = 15,
    ):
        assert isinstance(input_batch, (torch.Tensor, int, Sequence)), f'input_batch should be a torch.Tensor batch or integer representing the number of samples, or a sequence of the indices of the samples themselves, got: {repr(input_batch)}'
        assert log_wandb or log_local or save_dir
        assert isinstance(name, str) and name.strip()
        self._name = name
        self._count = 0
        self._every_n_steps = every_n_steps
        self._wandb = log_wandb
        self._local = log_local
        self._save_dir = save_dir
        self._figwidth = figwidth
        self._input_batch = input_batch
        self._mean_std = mean_std

    @rank_zero_only
    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch: torch.Tensor, batch_idx: int, dataloader_idx: int) -> None:
        self._count += 1
        if self._count % self._every_n_steps != 0:
            return

        # skip if nothing should be logged!
        if not (self._wandb or self._local or self._save_dir):
            return

        # get observations
        if isinstance(self._input_batch, torch.Tensor):
            xs = self._input_batch
            assert isinstance(xs, torch.Tensor)
            assert xs.ndim in (3, 4)
        else:  # (int, Sequence):
            dataset = trainer.train_dataloader.dataset
            assert isinstance(dataset, CombinedDataset)
            dataset = dataset.datasets
            if isinstance(dataset, IterableDataset):
                warnings.warn('IterableDataset`s are not supported!')
                return
            indices = np.random.randint(0, len(dataset), size=self._input_batch) if isinstance(self._input_batch, int) else self._input_batch
            xs = torch.stack([dataset[i] for i in indices], dim=0)
            assert isinstance(xs, torch.Tensor)
            assert xs.ndim == 4

        # generate grid
        with torch.no_grad(), evaluate_context(pl_module) as eval_module:
            # move to correct device
            xs = xs.to(eval_module.device)
            # feed forward -- TODO: handle different modes
            rs = eval_module(xs)
            # undo normalize
            if self._mean_std is not None:
                mean, std = self._mean_std
                xs = (xs * std) + mean  # revert normalize step: xs = (xs - mean) / std
                rs = (rs * std) + mean  # revert normalize step: rs = (rs - mean) / std
            # convert to uint8
            xs = torch.moveaxis(torch.clip(xs * 255, 0, 255).to(torch.uint8), 1, -1).detach().cpu().numpy()
            rs = torch.moveaxis(torch.clip(rs * 255, 0, 255).to(torch.uint8), 1, -1).detach().cpu().numpy()
            # make grid
            img = make_image_grid(np.concatenate([xs, rs]) if (xs.ndim == 4) else rs, num_cols=len(xs), pad=4)
            # add extra channels & check image size
            if img.shape[-1] == 1:
                img = np.dstack([img, img, img])
            assert img.ndim == 3
            assert img.shape[-1] == 3

        # plot online
        if self._wandb:
            import wandb
            wandb.log({self._name: wandb.Image(img)})
            logger.info('logged wandb model visualisation')

        # plot local
        if self._local:
            from matplotlib import pyplot as plt
            w, h = img.shape[:2]
            fig, ax = plt.subplots(figsize=(self._figwidth/w*h, self._figwidth))
            ax.imshow(img)
            ax.set_axis_off()
            fig.tight_layout()
            plt.show()
            logger.info('shown matplotlib model visualisation')

        # save locally
        if self._save_dir:
            import imageio
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir, exist_ok=True)
            path = os.path.abspath(os.path.join(self._save_dir, f'vis_{self._name}_{trainer.global_step}.png'))
            imageio.imsave(path, img)
            logger.info(f'saved model visualisation to: {repr(path)}')


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
