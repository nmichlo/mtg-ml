import logging
from typing import Optional
from typing import Sequence

import numpy as np
import torch


logger = logging.getLogger(__name__)


import torch.nn.functional as F
import disent.dataset.transform.functional as F_d
from disent.dataset.transform import ToImgTensorF32 as _ToImgTensorF32


# ========================================================================= #
# Transform                                                                 #
# ========================================================================= #


def _pad_tensor_img_to_square(img_tensor: torch.Tensor):
    # TODO: move this into disent?
    assert img_tensor.ndim in (2, 3)
    # get width and height from (C, H, W) or (H, W)
    (*_, H, W) = img_tensor.shape
    # only pad if we need to
    if H != W:
        gap = abs(W - H) / 2
        pad_l = int(np.floor(gap))
        pad_r = int(np.ceil(gap))
        # handle different cases
        if H < W:
            pad = (0, 0, pad_l, pad_r)  # from last to first dim [Wl, Wr, Hl, Hr, Cl, Cr]
        else:
            pad = (pad_l, pad_r, 0, 0)  # from last to first dim [Wl, Wr, Hl, Hr, Cl, Cr]
        # pad the observation
        img_tensor = F.pad(img_tensor, pad=pad, mode='constant', value=0)
    # done
    return img_tensor


def _tuple_if_iter_else_float(val):
    # TODO: move this into disent
    try:
        return tuple(val)
    except:
        return float(val)


class ToStandardF32(_ToImgTensorF32):

    def __init__(
        self,
        size: Optional[F_d.SizeType] = None,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        pad_to_square: bool = True,
    ):
        super().__init__(size=size, mean=None, std=None)
        self._mean = _tuple_if_iter_else_float(mean) if (mean is not None) else None
        self._std = _tuple_if_iter_else_float(std) if (std is not None) else None
        self._pad_to_square = pad_to_square

    def __call__(self, obs) -> torch.Tensor:
        obs = super().__call__(obs)
        # pad if specified
        if self._pad_to_square:
            obs = _pad_tensor_img_to_square(obs)
        # done!
        return obs

# ========================================================================= #
# Get Obs                                                                   #
# ========================================================================= #


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
