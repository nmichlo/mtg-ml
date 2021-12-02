import os.path

import hydra
import pytorch_lightning as pl
import torch

from mtg_ml.util.func import instantiate_required


def run_training(cfg):
    # create system
    system = instantiate_required(cfg.system.system_cls, instance_of=pl.LightningModule)
    # create data loader
    datamodule = instantiate_required(cfg.system.datamodule_cls, instance_of=pl.LightningDataModule)
    # create trainer
    trainer = pl.Trainer(
        max_steps=cfg.system.system_cls.lr_anneal_steps,
        gpus=1 if torch.cuda.is_available() else 0,
    )
    # train
    trainer.fit(system, datamodule)



# @dataclass
# class _TrainCfg(object):
#     data_dir: str = ""
#     lr: float = 1e-4
#     weight_decay: float = 0.0
#     lr_anneal_steps: int = 0
#     batch_size: int = 1
#     # microbatch: int = -1  # -1 disables microbatches
#     ema_rate: Optional[float] = 0.9999
#     # log_interval: int = 10
#     # save_interval: int = 10000
#     # resume_checkpoint: str = ""
#     # use_fp16: bool = False
#     # fp16_scale_growth: float = 1e-3


if __name__ == "__main__":

    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    CONFIG_NAME = 'config'

    @hydra.main(CONFIG_PATH, CONFIG_NAME)
    def main(cfg):
        run_training(cfg)

    main()
