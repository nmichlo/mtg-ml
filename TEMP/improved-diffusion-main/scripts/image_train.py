"""
Train a diffusion model on images.
"""

from dataclasses import dataclass

from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import ImageModelDiffusionCfg, create_model_and_diffusion
from improved_diffusion.train_util import TrainLoop


def main():
    cfg = ImageTrainCfg.parse_args()

    # create model and diffusion
    model, diffusion = create_model_and_diffusion(cfg)
    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)

    # create data loader
    data = load_data(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        class_cond=cfg.class_cond,
    )

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


@dataclass
class ImageTrainCfg(ImageModelDiffusionCfg):
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


if __name__ == "__main__":
    main()
