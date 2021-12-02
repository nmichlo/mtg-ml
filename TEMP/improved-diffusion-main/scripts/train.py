import os.path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from improved_diffusion.image_datasets import load_data
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
