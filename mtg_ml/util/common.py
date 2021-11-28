import logging
import os
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms
from disent.dataset.data import Hdf5Dataset
from disent.util.visualize.vis_util import make_image_grid
from matplotlib import pyplot as plt
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Normalize


logger = logging.getLogger(__name__)


# ========================================================================= #
# Transform                                                                 #
# ========================================================================= #


def seed(seed):
    if seed is None:
        warnings.warn('No random seed set!')
        return
    # import components
    import random
    import numpy as np
    import torch
    # seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('[seeded]:', seed)


# ========================================================================= #
# Transform                                                                 #
# ========================================================================= #


# move into disent?
def _count_params(model, trainable=None):
    if model is None:
        return 0
    if trainable is None:
        return sum(p.numel() for p in model.parameters())
    elif trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)


# move into disent?
def count_params(model, trainable=None):
    p = _count_params(model, trainable)
    pow = 0 if (p == 0) else int(np.log(p) / np.log(1000))
    mul = 1000 ** pow
    symbol = {0: '', 1: 'K', 2: 'M', 3: 'B', 4: 'T', 5: 'P', 6: 'E'}[pow]
    return f'{p/mul:5.1f}{symbol}'


# ========================================================================= #
# Transform                                                                 #
# ========================================================================= #


class ToTensor(object):
    def __init__(self, move_channels=True):
        self._move_channels = move_channels

    def __call__(self, img):
        if self._move_channels:
            img = np.moveaxis(img, -1, -3)
        img = img.astype('float32') / 255
        img = torch.from_numpy(img)
        return img


class MlSystem(pl.LightningModule):

    def get_progress_bar_dict(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    # override these .. we should rather use a DataModule!
    train_dataloader = None
    test_dataloader = None
    val_dataloader = None
    predict_dataloader = None


def get_image_obs(obs) -> torch.Tensor:
    # get batch from datasets with labels
    if not isinstance(obs, torch.Tensor):
        obs = obs[0]
    # add missing channel
    if obs.ndim == 2:
        obs = obs[None, :, :]
    # check
    assert isinstance(obs, torch.Tensor)
    assert obs.ndim == 3
    return obs


def get_image_batch(batch) -> torch.Tensor:
        # get batch from datasets with labels
        if not isinstance(batch, torch.Tensor):
            batch = batch[0]
        # add missing channel
        if batch.ndim == 3:
            batch = batch[:, None, :, :]
        # check
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 4
        return batch


# ========================================================================= #
# Visualise                                                                 #
# ========================================================================= #


# @lru_cache()
# def _fn_has_param(fn, param: str):
#     import inspect
#     return param in inspect.signature(fn).parameters


# ========================================================================= #
# Hdf5 Data Module                                                          #
# ========================================================================= #


class Hdf5DataModule(pl.LightningDataModule):

    def __init__(self, h5_path: str, h5_dataset_name: str = 'data', batch_size: int = 64, val_ratio: float = 0.1, num_workers: int = os.cpu_count(), to_tensor=True, mean_std: Tuple[float, float] = None, transform=None, in_memory: bool = False):
        super().__init__()
        self._batch_size = batch_size
        self._val_ratio = val_ratio
        self._num_workers = num_workers
        # get transforms
        transforms = []
        if to_tensor: transforms.append(ToTensor(move_channels=True))
        if mean_std: transforms.append(Normalize(*mean_std))
        if transform: transforms.append(transform)
        # load h5py data
        self._data = Hdf5Dataset(
            h5_path=h5_path,
            h5_dataset_name=h5_dataset_name,
            transform=torchvision.transforms.Compose(transforms),
        )
        # load into memory
        if in_memory:
            print('loading into memory...', end=' ')
            self._data = self._data.numpy_dataset()
            print('done loading!')
        # self.dims is returned when you call dm.size()
        self.dims = self._data.shape[1:]

    @property
    def data(self):
        return self._data

    def setup(self, stage: Optional[str] = None):
        self._data_trn, self._data_val = random_split(
            dataset=self._data,
            lengths=[
                int(np.floor(len(self._data) * (1 - self._val_ratio))),
                int(np.ceil(len(self._data) * self._val_ratio)),
            ],
        )

    def train_dataloader(self):
        return DataLoader(dataset=self._data_trn, num_workers=self._num_workers, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self._data_val, num_workers=self._num_workers, batch_size=self._batch_size, shuffle=True)


# ========================================================================= #
# Wandb Setup & Finish Callback                                             #
# ========================================================================= #


class WandbContextManagerCallback(pl.Callback):

    def __init__(self, extra_entries: dict = None):
        self._extra_entries = {} if (extra_entries is None) else extra_entries

    @rank_zero_only
    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
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
        wandb.finish()


# ========================================================================= #
# makers                                                                    #
# ========================================================================= #


@contextmanager
def evaluate_context(module: torch.nn.Module, train: bool = False):
    """
    Temporarily switch a model to evaluation
    mode, and restore the mode afterwards!
    """
    was_training = module.training
    try:
        module.train(mode=train)
        yield module
    finally:
        module.train(mode=was_training)


# ========================================================================= #
# makers                                                                    #
# ========================================================================= #


# def make_mtg_datamodule(
#     batch_size: int = 32,
#     num_workers: int = os.cpu_count(),
#     val_ratio: float = 0,
#     # convert options
#     load_path: str = None,
#     data_root: Optional[str] = None,
#     convert_kwargs: Dict[str, Any] = None,
#     # extra data_loader transform
#     to_tensor=True,
#     transform=None,
#     mean_std: Optional[Tuple[float, float]] = None,
#     # memory
#     in_memory=False,
# ):
#     from mtgdata.scryfall_convert import generate_converted_dataset
#
#     # generate training set
#     if load_path is None:
#         if convert_kwargs is None:
#             convert_kwargs = {}
#         h5_path, meta_path = generate_converted_dataset(save_root=data_root, data_root=data_root, **convert_kwargs)
#     else:
#         assert not convert_kwargs, '`convert_kwargs` cannot be set if `data_path` is specified'
#         assert not data_root, '`data_root` cannot be set if `data_path` is specified'
#         h5_path = load_path
#
#     # get transform
#
#     return Hdf5DataModule(
#         h5_path,
#         batch_size=batch_size,
#         val_ratio=val_ratio,
#         num_workers=num_workers,
#         to_tensor=to_tensor,
#         mean_std=mean_std,
#         transform=transform,
#         in_memory=in_memory,
#     )

#
# def make_mtg_trainer(
#     # training
#     train_epochs: int = None,
#     train_steps: int = None,
#     cuda: Union[bool, int] = torch.cuda.is_available(),
#     # visualise
#     visualize_period: int = 500,
#     visualize_input: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[float, float]]]]] = None,
#     # utils
#     checkpoint_period: int = 2500,
#     checkpoint_dir: str = 'checkpoints',
#     checkpoint_monitor: Optional[str] = 'loss',
#     resume_from_checkpoint: str = None,
#     # trainer kwargs
#     trainer_kwargs: dict = None,
#     # logging
#     wandb_enabled: bool = False,
#     wandb_name: str = None,
#     wandb_project: str = None,
#     wandb_kwargs: dict = None,
# ):
#     time_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
#
#     # initialise callbacks
#     callbacks = []
#     if wandb_enabled:
#         callbacks.append(WandbContextManagerCallback())
#     if visualize_period and (visualize_input is not None):
#         for k, v in visualize_input.items():
#             v, mean_std = (v if isinstance(v, tuple) else (v, None))
#             callbacks.append(VisualiseCallback(name=k, input_batch=v, every_n_steps=visualize_period, log_wandb=wandb_enabled, log_local=not wandb_enabled, mean_std=mean_std))
#
#     if checkpoint_period:
#         from pytorch_lightning.callbacks import ModelCheckpoint
#         callbacks.append(ModelCheckpoint(
#             dirpath=os.path.join(checkpoint_dir, time_str),
#             monitor=checkpoint_monitor,
#             every_n_train_steps=checkpoint_period,
#             verbose=True,
#             save_top_k=None if (checkpoint_monitor is None) else 5,
#         ))
#
#     # initialise logger
#     logger = True
#     if wandb_enabled:
#         assert isinstance(wandb_name, str) and wandb_name, f'`wandb_name` must be a non-empty str, got: {repr(wandb_name)}'
#         assert isinstance(wandb_project, str) and wandb_project, f'`wandb_project` must be a non-empty str, got: {repr(wandb_project)}'
#         from pytorch_lightning.loggers import WandbLogger
#         logger = WandbLogger(name=f'{time_str}:{wandb_name}', project=wandb_project, **(wandb_kwargs if (wandb_kwargs is not None) else {}))
#
#     # initialise model trainer
#     return pl.Trainer(
#         gpus=(1 if cuda else 0) if isinstance(cuda, bool) else cuda,
#         max_epochs=train_epochs,
#         max_steps=train_steps,
#         # checkpoint_callback=False,
#         logger=logger,
#         resume_from_checkpoint=resume_from_checkpoint,
#         callbacks=callbacks,
#         weights_summary='full',
#         # extra kwargs
#         **(trainer_kwargs if trainer_kwargs else {}),
#     )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
