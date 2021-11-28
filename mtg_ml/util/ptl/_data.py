import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from disent.dataset.data import Hdf5Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from mtg_ml.util.func import instantiate_or_none


logger = logging.getLogger(__name__)


# ========================================================================= #
# Hdf5 Data Module                                                          #
# ========================================================================= #


class Hdf5DataModule(pl.LightningDataModule):

    test_dataloader = None
    predict_dataloader = None

    def __init__(
        self,
        h5_path: str,
        h5_dataset_name: str = 'data',
        batch_size: int = 64,
        val_ratio: float = 0.1,
        split_seed: int = 7777,
        num_workers: int = min(os.cpu_count(), 8),
        in_memory: bool = False,
        transform: Optional[dict] = None,
    ):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # save variables
        assert 0 <= self.hparams.val_ratio < 1
        # make transform
        transform = instantiate_or_none(self.hparams.transform)
        assert (transform is None) or callable(transform), f'transform must be callable or None, got: {repr(transform)}'
        # load h5py data
        self._data = Hdf5Dataset(
            h5_path=self.hparams.h5_path,
            h5_dataset_name=self.hparams.h5_dataset_name,
            transform=transform,
        )
        # load into memory
        if self.hparams.in_memory:
            logger.info('loading into memory...', end=' ')
            self._data = self._data.numpy_dataset()
            logger.info('done loading!')
        # self.dims is returned when you call dm.size()
        self.dims = self._data.shape[1:]

    @property
    def data(self):
        return self._data

    def setup(self, stage: Optional[str] = None):
        self._data_trn, self._data_val = random_split(
            dataset=self._data,
            lengths=[
                int(np.floor(len(self._data) * (1 - self.hparams.val_ratio))),
                int(np.ceil(len(self._data) * self.hparams.val_ratio)),
            ],
            generator=torch.Generator().manual_seed(self.hparams.split_seed)
        )

    def train_dataloader(self):
        return DataLoader(dataset=self._data_trn, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self._data_val, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, shuffle=True)



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
