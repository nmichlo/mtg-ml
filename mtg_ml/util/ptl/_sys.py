import pytorch_lightning as pl


# ========================================================================= #
# Base System With Sane Defaults                                            #
# ========================================================================= #


class MlSystem(pl.LightningModule):

    def get_progress_bar_dict(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    # override these .. we should rather use a DataModule!
    train_dataloader = None
    test_dataloader = None
    val_dataloader = None
    predict_dataloader = None


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
