import os.path
import hydra
import pytorch_lightning as pl
from mtg_ml.util.func import instantiate_required


# TODO: we have not implemented: scripts/image_nll.py


def run_training(cfg):
    # create system
    system = instantiate_required(cfg.system.system_cls, instance_of=pl.LightningModule)
    # create data loader
    datamodule = instantiate_required(cfg.system.datamodule_cls, instance_of=pl.LightningDataModule)
    # create trainer
    trainer = instantiate_required(cfg.system.trainer, pl.Trainer)
    # train
    trainer.fit(system, datamodule)


if __name__ == "__main__":

    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    CONFIG_NAME = 'config'

    @hydra.main(CONFIG_PATH, CONFIG_NAME)
    def main(cfg):
        run_training(cfg)

    main()
