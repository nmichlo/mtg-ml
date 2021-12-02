import argparse
from dataclasses import dataclass
from typing import Union

import torch.nn.functional as F

from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import create_model_and_diffusion
from improved_diffusion.script_util import ImageModelDiffusionCfg
from improved_diffusion.script_util import SrModelDiffusionCfg
from improved_diffusion.train_util import TrainLoop


def run_training(cfg: Union['ImageTrainCfg', 'SrTrainCfg']):
    # create model and diffusion
    model, diffusion = create_model_and_diffusion(cfg)
    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)

    # create data loader
    if isinstance(cfg, ImageTrainCfg):
        data = load_data(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            class_cond=cfg.class_cond,
        )
    elif isinstance(cfg, SrTrainCfg):
        data = load_superres_data(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            large_size=cfg.image_size,
            small_size=cfg.small_size,
            class_cond=cfg.class_cond,
        )
    else:
        raise TypeError(f'cfg must be of type {ImageTrainCfg.__name__} or {SrTrainCfg.__name__}, got: {type(cfg)}')

    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg.batch_size,
        microbatch=cfg.microbatch,
        lr=cfg.lr,
        ema_rate=cfg.ema_rate,
        # log_interval=cfg.log_interval,
        # save_interval=cfg.save_interval,
        # resume_checkpoint=cfg.resume_checkpoint,
        # use_fp16=cfg.use_fp16,
        # fp16_scale_growth=cfg.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.weight_decay,
        lr_anneal_steps=cfg.lr_anneal_steps,
    )

    # train
    trainer.run_loop()


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
    schedule_sampler: str = "uniform"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_anneal_steps: int = 0
    batch_size: int = 1
    microbatch: int = -1  # -1 disables microbatches
    ema_rate: str = "0.9999"  # comma-separated list of EMA values
    # log_interval: int = 10
    # save_interval: int = 10000
    # resume_checkpoint: str = ""
    # use_fp16: bool = False
    # fp16_scale_growth: float = 1e-3


@dataclass
class ImageTrainCfg(ImageModelDiffusionCfg, _TrainCfg):
    pass


@dataclass
class SrTrainCfg(SrModelDiffusionCfg, _TrainCfg):
    pass


if __name__ == "__main__":
    # subcommands
    cli = argparse.ArgumentParser()
    parsers = cli.add_subparsers()
    parsers.required = True

    # subcommand: train images
    parser_im = parsers.add_parser('im')
    parser_im.set_defaults(_cfg_cls_=ImageTrainCfg)
    ImageTrainCfg.add_parser_args(parser=parser_im)

    # subcommand: train sr
    parser_sr = parsers.add_parser('sr')
    parser_sr.set_defaults(_cfg_cls_=SrTrainCfg)
    SrTrainCfg.add_parser_args(parser=parser_sr)

    # run the specified subcommand!
    args = cli.parse_args()
    # get cfg
    cfg = args._cfg_cls_
    del args._cfg_cls_
    cfg = cfg(**args.__dict__)
    # run model
    run_training(cfg)
