#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

from typing import Optional

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mtg_ml.nn.model_alt import AutoEncoderSkips
from mtg_ml.nn.loss import get_recon_loss
from mtg_ml.nn.loss import kl_div
from mtg_ml.util.ptl.sys import MlSystem


# ========================================================================= #
# System                                                                    #
# ========================================================================= #


class MtgVaeSystem(MlSystem):

    def __init__(
        self,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 0.003,
        # model options
        z_size: int = 128,
        repr_hidden_size: Optional[int] = None,
        repr_channels: int = 16,
        channel_mul: float = 1.5,
        channel_start: int = 16,
        channel_dec_mul: float = 1.0,
        model_activation: str = 'leaky_relu',
        model_norm: Optional[str] = None,
        model_weight_init: Optional[str] = None,
        model_skip_mode='next',
        model_skip_downsample: str = 'ave',     # max, ave
        model_skip_upsample: str = 'bilinear',  # nearest, bilinear
        model_downsample: str = 'stride',       # stride, max, ave
        model_upsample: str = 'stride',         # stride, nearest, bilinear
        # loss options
        recon_loss: str = 'mse',
    ):
        super().__init__()
        self.save_hyperparameters()
        # make model
        self.model = AutoEncoderSkips(
            # sizes
            z_size=self.hparams.z_size,
            repr_hidden_size=self.hparams.repr_hidden_size,
            c_repr=self.hparams.repr_channels,
            channel_mul=self.hparams.channel_mul,
            channel_start=self.hparams.channel_start,
            channel_dec_mul=self.hparams.channel_dec_mul,
            # layers
            weight_init=self.hparams.model_weight_init,
            activation=self.hparams.model_activation,
            norm=self.hparams.model_norm,
            skip_mode=self.hparams.model_skip_mode,
            skip_upsample=self.hparams.model_skip_upsample,
            skip_downsample=self.hparams.model_skip_downsample,
            upsample=self.hparams.model_upsample,
            downsample=self.hparams.model_downsample,
            sigmoid_out=False,
        )
        # get loss
        self._loss = get_recon_loss(self.hparams.recon_loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.25,
            patience=10,
            verbose=True,
            min_lr=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "recon",
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        recon, z, posterior, prior = self.model.forward_train(batch)
        # compute recon loss
        loss_recon = self.hparams.alpha * self._loss(recon, batch, reduction='mean')
        # compute kl divergence
        loss_kl = self.hparams.beta * kl_div(posterior, prior).mean()
        # combined loss
        loss = loss_recon + loss_kl
        # return loss
        self.log('kl',    loss_kl,    on_step=True, prog_bar=True)
        self.log('recon', loss_recon, on_step=True)
        self.log('loss',  loss,       on_step=True)
        return loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
