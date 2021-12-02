"""
Train a super-resolution model.
"""
from dataclasses import dataclass

import torch.nn.functional as F

from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import SrModelDiffusionCfg, create_model_and_diffusion
from improved_diffusion.train_util import TrainLoop


def main():
    args = SrTrainCfg.parse_args()

    # create model and diffusion
    model, diffusion = create_model_and_diffusion(args)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # create data loader
    data = load_superres_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        large_size=args.image_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )

    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        # log_interval=args.log_interval,
        # save_interval=args.save_interval,
        # resume_checkpoint=args.resume_checkpoint,
        # use_fp16=args.use_fp16,
        # fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
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
class SrTrainCfg(SrModelDiffusionCfg):
    data_dir: str = ""
    schedule_sampler: str = "uniform"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_anneal_steps: int = 0
    batch_size: int = 1
    microbatch: int = -1
    ema_rate: str = "0.9999"
    # log_interval: int = 10
    # save_interval: int = 10000
    # resume_checkpoint: str = ""
    # use_fp16: bool = False
    # fp16_scale_growth: float = 1e-3


if __name__ == "__main__":
    main()
