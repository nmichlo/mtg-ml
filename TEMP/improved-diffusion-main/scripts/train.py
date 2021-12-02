import argparse
import os.path
from dataclasses import dataclass
from typing import Optional
from typing import Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import create_diffusion_and_sampler
from improved_diffusion.script_util import create_model
from improved_diffusion.script_util import ImageModelCfg
from improved_diffusion.script_util import SrModelCfg
from improved_diffusion.script_util import DiffusionAndSampleCfg
from improved_diffusion.train_util import IDDPM
from mtg_ml.util.func import instantiate_required


class ImDataModule(pl.LightningDataModule):

    test_dataloader = None
    val_dataloader = None
    predict_dataloader = None

    def __init__(self):
        super().__init__()
        self._data = load_data(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            class_cond=cfg.class_cond,
        )

    def train_dataloader(self):
        pass


class SrDataModule(ImDataModule):

    def __init__(self):
        super().__init__()
        self._data = load_superres_data(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            large_size=cfg.image_size,
            small_size=cfg.small_size,
            class_cond=cfg.class_cond,
        )



def run_training(cfg):

    # create data loader
    datamodule = instantiate_required(cfg.datamodule, instance_of=(ImDataModule, SrDataModule))

    system = IDDPM(
        # targets
        model=cfg.system.cfg_model,
        diffusion=cfg.system.cfg_diffusion_and_sample,
        # hparams
        lr=cfg.system.lr,
        lr_anneal_steps=cfg.system.lr_anneal_steps,
        weight_decay=cfg.system.weight_decay,
        ewm_rate=cfg.system.ema_rate,
    )

    trainer = pl.Trainer(
        max_steps=cfg.system.lr_anneal_steps,
        gpus=1 if torch.cuda.is_available() else 0,
    )

    # train
    trainer.fit(system, datamodule)


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs


@dataclass
class _TrainCfg(object):
    data_dir: str = ""
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_anneal_steps: int = 0
    batch_size: int = 1
    # microbatch: int = -1  # -1 disables microbatches
    ema_rate: Optional[float] = 0.9999
    # log_interval: int = 10
    # save_interval: int = 10000
    # resume_checkpoint: str = ""
    # use_fp16: bool = False
    # fp16_scale_growth: float = 1e-3


if __name__ == "__main__":

    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    CONFIG_NAME = 'config'

    @hydra.main(CONFIG_PATH, CONFIG_NAME)
    def main(cfg):
        run_training(cfg)

    main()

    # trainer = TrainLoop(
    #     model=model,
    #     diffusion=diffusion,
    #     data=data,
    #     #
    #     # batch_size=params.batch_size, # TODO: add to data
    #     # microbatch=params.microbatch,
    #     lr=params.lr,
    #     ema_rate=params.ema_rate,
    #     # log_interval=params.log_interval,
    #     # save_interval=params.save_interval,
    #     # resume_checkpoint=params.resume_checkpoint,
    #     # use_fp16=params.use_fp16,
    #     # fp16_scale_growth=params.fp16_scale_growth,
    #     schedule_sampler=schedule_sampler,
    #     weight_decay=params.weight_decay,
    #     lr_anneal_steps=params.lr_anneal_steps,
    # )
