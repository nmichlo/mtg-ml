"""
Converted to pytorch_lightning from:
https://github.com/taldatech/soft-intro-vae-pytorch
"""

import dataclasses
import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from examples.common import BaseLightningModule
from examples.common import make_mtg_datamodule
from examples.common import make_mtg_trainer
from examples.common import normalize_image_batch
from examples.common import normalize_image_obs
from examples.experiments.nn_weights import init_model_weights
from examples.experiments.nn_weights import init_weights
from examples.nn.loss import get_recon_loss
from examples.nn.loss import kl_loss
from examples.nn.model import BaseAutoEncoder


# ========================================================================= #
# Model                                                                     #
# ========================================================================= #
from examples.nn.model_alt import AutoEncoderSkips


class ResidualBlock(nn.Module):
    """
    ORIG: https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)

    FROM: https://github.com/taldatech/soft-intro-vae-pytorch
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
            self.conv_expand = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1   = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2   = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        skip = self.conv_expand(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)) + skip)
        return x


class Encoder(nn.Module):

    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256):
        super(Encoder, self).__init__()

        self.cdim = cdim
        self.image_size = image_size
        cc = channels[0]

        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2
        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))

        # compute input shape to fcn | feed forward data!
        self.conv_output_size = self.main(torch.zeros(1, cdim, image_size, image_size)).shape[1:]
        num_fc_features       = int(np.prod(self.conv_output_size))

        # final layer
        self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        return y


class Decoder(nn.Module):

    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conv_input_size=None):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]

        # TODO: I shouldn't have to pass this
        assert conv_input_size is not None
        self.conv_input_size = conv_input_size
        num_fc_features = int(np.prod(conv_input_size))

        self.fc = nn.Sequential(
            nn.Linear(zdim, num_fc_features),
            nn.ReLU(True),
        )

        sz = 4
        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


# ========================================================================= #
# VAE                                                                       #
# ========================================================================= #


class SoftIntroVae(BaseAutoEncoder):

    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256):
        super(SoftIntroVae, self).__init__()
        self.z_dim = zdim
        self._encoder = Encoder(cdim, zdim, channels, image_size)
        self._decoder = Decoder(cdim, zdim, channels, image_size, conv_input_size=self._encoder.conv_output_size)

    def _enc(self, x):
        return self._encoder(x)

    def _dec(self, z):
        return self._decoder(z)


# ========================================================================= #
# Training                                                                  #
# ========================================================================= #


_DATA_ROOT = 'data'


@dataclasses.dataclass
class DatasetSettings:
    image_size: int
    ch: int
    channels: Tuple[int, ...]
    make_dataset: callable


def _get_celeba(image_size: int, train_size: int):
    from torchvision.datasets import CelebA, ImageFolder
    import os
    # verify celeba | manual download from: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
    data = CelebA(_DATA_ROOT, download=False)
    # load images directly
    data = ImageFolder(os.path.join(data.root, data.base_folder), transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))
    # original partition code: [x for x in os.listdir('data/celeba/img_align_celeba') if is_image_file(x)][:162770]
    data = torch.utils.data.Subset(data, indices=list(range(train_size)))
    return data


_DATASET_SETTINGS = {
    'cifar10':     DatasetSettings(image_size=32,   ch=3, channels=(64, 128, 256),                        make_dataset=lambda:     torchvision.datasets.CIFAR10(root=_DATA_ROOT, train=True,     download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))),
    'celeb128':    DatasetSettings(image_size=128,  ch=3, channels=(64, 128, 256, 512, 512),              make_dataset=lambda: _get_celeba(1024, 162770)),
    'celeb256':    DatasetSettings(image_size=256,  ch=3, channels=(64, 128, 256, 512, 512, 512),         make_dataset=lambda: _get_celeba(1024, 162770)),
    'celeb1024':   DatasetSettings(image_size=1024, ch=3, channels=(16, 32, 64, 128, 256, 512, 512, 512), make_dataset=lambda: _get_celeba(1024, 29000)),
  # 'monsters128': DatasetSettings(image_size=128,  ch=3, channels=(64, 128, 256, 512, 512),              make_dataset=lambda:            DigitalMonstersDataset(root_path=os.path.join(_DATA_ROOT, 'monsters'), output_height=128 , transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))),
    'svhn':        DatasetSettings(image_size=32,   ch=3, channels=(64, 128, 256),                        make_dataset=lambda:         torchvision.datasets.SVHN(root=_DATA_ROOT, split='train', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))),
    'fmnist':      DatasetSettings(image_size=28,   ch=1, channels=(64, 128),                             make_dataset=lambda: torchvision.datasets.FashionMNIST(root=_DATA_ROOT, train=True,    download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))),
    'mnist':       DatasetSettings(image_size=28,   ch=1, channels=(64, 128),                             make_dataset=lambda:        torchvision.datasets.MNIST(root=_DATA_ROOT, train=True,    download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))),
}


def get_dataset_settings(dataset: str) -> DatasetSettings:
    return _DATASET_SETTINGS[dataset]


# ========================================================================= #
# Training                                                                  #
# ========================================================================= #


class SoftIntroVaeSystem(BaseLightningModule):

    def __init__(
        self,
        # model
            dataset: str,
            z_dim: int = 128,
        # training
            lr_enc: float = 2e-4,
            lr_dec: float = 2e-4,
            lr_vae: float = 1e-3,
            recon_loss: str = "mse",
            train_steps_vae: int = 0,
        # loss scaling
            beta_kl: float = 1.0,
            beta_rec: float = 1.0,
            beta_neg: float = 256.0,
            gamma_r: float = 1e-8,
        # unimplemented:
            # exit_on_negative_diff: bool = False, TODO: implement
            # training_measure_fid: bool = False,  # TODO: implement
    ):
        super().__init__()
        self.save_hyperparameters()
        # initialise
        self.dataset_settings = get_dataset_settings(self.hparams.dataset)
        self._scale = 1 / (self.dataset_settings.ch * self.dataset_settings.image_size**2)  # 1 / (C * H * W)
        self._loss = get_recon_loss(self.hparams.recon_loss)
        # make model
        self.model = SoftIntroVae(cdim=self.dataset_settings.ch, zdim=self.hparams.z_dim, channels=self.dataset_settings.channels, image_size=self.dataset_settings.image_size)

    def configure_optimizers(self):
        optimizer_vae = optim.Adam(self.model.parameters(),          lr=self.hparams.lr_vae)
        optimizer_enc = optim.Adam(self.model._encoder.parameters(), lr=self.hparams.lr_enc)
        optimizer_dec = optim.Adam(self.model._decoder.parameters(), lr=self.hparams.lr_dec)
        vae_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_vae, milestones=(350,), gamma=0.1)
        enc_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_enc, milestones=(350,), gamma=0.1)
        dec_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_dec, milestones=(350,), gamma=0.1)
        return [optimizer_vae, optimizer_enc, optimizer_dec], [vae_scheduler, enc_scheduler, dec_scheduler]

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        # always 4 dimensions & a single tensor
        batch = normalize_image_batch(batch)

        # CALCULATE FID SCORE
        self._compute_fid()

        # STANDARD VAE TRAINING STEP (update encoder and decoder)
        # - this should be the same as the logic in train_simple_vae
        if self.trainer.global_step < self.hparams.train_steps_vae:
            # skip other steps
            if optimizer_idx != 0:
                return None
            # train vae like usual
            recon, z, posterior, prior = self.model.forward_train(batch)
            # compute recon loss
            loss_recon = self.hparams.beta_rec * self._loss(recon, batch, reduction='mean')
            # compute kl divergence
            loss_kl = self.hparams.beta_kl * kl_loss(posterior, prior, reduction='mean')
            # combined loss
            loss = loss_recon + loss_kl
            # return loss
            self.log('vae_kl',    loss_kl,    on_step=True, prog_bar=True)
            self.log('vae_recon', loss_recon, on_step=True)
            self.log('vae_loss',  loss,       on_step=True)
            return loss

        # ENCODER TRAINING STEP
        if optimizer_idx == 1:
            recon_fake = self.model.decode(torch.randn(size=(len(batch), self.hparams.z_dim), device=self.device))

            # feed forward
            rec_real, z_real, posterior_real, prior_real = self.model.forward_train(batch)
            rec_rec,  z_rec,  posterior_rec,  prior_rec  = self.model.forward_train(rec_real.detach())
            rec_fake, z_fake, posterior_fake, prior_fake = self.model.forward_train(recon_fake.detach())

            # reconstruction losses
            loss_real = self._loss(rec_real, batch, reduction="mean")
            loss_rec  = torch.flatten(self._loss(rec_rec,  rec_real,   reduction='none'), start_dim=1).sum(dim=-1)
            loss_fake = torch.flatten(self._loss(rec_fake, recon_fake, reduction='none'), start_dim=1).sum(dim=-1)

            # kl divergences
            kl_real = kl_loss(posterior_real, prior_real, reduction="mean")
            kl_rec  = kl_loss(posterior_rec,  prior_rec,  reduction="none").sum(dim=-1)
            kl_fake = kl_loss(posterior_fake, prior_fake, reduction="none").sum(dim=-1)

            # exp elbo
            expelbo_rec  = (-2 * self._scale * (self.hparams.beta_rec * loss_rec  + self.hparams.beta_neg * kl_rec)).exp().mean()
            expelbo_fake = (-2 * self._scale * (self.hparams.beta_rec * loss_fake + self.hparams.beta_neg * kl_fake)).exp().mean()

            # compute final loss
            elbo_real = self._scale * (self.hparams.beta_rec * loss_real + self.hparams.beta_kl * kl_real)
            elbo_fake = 0.25 * (expelbo_rec + expelbo_fake)
            loss      = elbo_real + elbo_fake

            # log everything
            with torch.no_grad():
                self.log('enc_loss',          loss,          prog_bar=False)
                self.log('enc_kl_real',       kl_real,       prog_bar=False)
                self.log('enc_rec_real',      loss_real,     prog_bar=False)
                self.log('enc_exp_elbo_rec',  expelbo_rec,   prog_bar=False)
                self.log('enc_exp_elbo_fake', expelbo_fake,  prog_bar=False)

            # backprop (must only update encoder)
            return loss

        elif optimizer_idx == 2:
            # this should originally be re-used from the previous step
            recon_fake = self.model.decode(torch.randn(size=(len(batch), self.hparams.z_dim), device=self.device))

            # feed forward
            rec_real, z_real, posterior_real, prior_real = self.model.forward_train(batch, detach_z=True)
            rec_rec,  z_rec,  posterior_rec,  prior_rec  = self.model.forward_train(rec_real, detach_z=True)
            rec_fake, z_fake, posterior_fake, prior_fake = self.model.forward_train(recon_fake, detach_z=True)

            # kl divergences
            loss_real = self._loss(rec_real, batch,               reduction="mean")
            loss_rec  = self._loss(rec_rec,  rec_real.detach(),   reduction="mean")
            loss_fake = self._loss(rec_fake, recon_fake.detach(), reduction="mean")

            # kl divergences
            kl_rec  = kl_loss(posterior_rec,  prior_rec,  reduction="mean")
            kl_fake = kl_loss(posterior_fake, prior_fake, reduction="mean")

            loss = self._scale * (
                    self.hparams.beta_rec * loss_real
                    + (self.hparams.beta_kl * 0.5) * (kl_rec + kl_fake)
                    + (self.hparams.gamma_r * self.hparams.beta_rec * 0.5) * (loss_rec + loss_fake)
            )

            # log everything
            with torch.no_grad():
                self.log('dec_loss',     loss,      prog_bar=False)
                self.log('dec_kl_fake',  kl_fake,   prog_bar=False)
                self.log('dec_kl_rec',   kl_rec,    prog_bar=False)
                self.log('dec_rec_real', loss_real, prog_bar=False)

            # backprop (must only update decoder)
            return loss

    def forward(self, x):
        return self.model(x)

    def _compute_fid(self, force=False):
        pass
        # if not self.hparams.training_measure_fid:
        #     return
        # # skip
        # if not force:
        #     if not ((self.trainer.global_step == 0) or (self.trainer.global_step % 1000 == 0)):
        #         return
        # # compute
        # with torch.no_grad():
        #     calculate_fid_given_dataset(self.trainer.train_data_loader, self.model, self.batch_size, cuda=True, dims=2048, device=self.device, num_images=50000)


# ========================================================================= #
# Training                                                                  #
# ========================================================================= #


if __name__ == '__main__':

    @dataclasses.dataclass
    class ModelSettings:
        beta_kl: float
        beta_rec: float
        beta_neg: float
        z_dim: int
        batch_size: int

    _SETTINGS = {
        'cifar10':     ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_dim=128, batch_size=32),
        'celeb128':    ModelSettings(beta_kl=1.0, beta_rec=0.5, beta_neg=1024, z_dim=256, batch_size=8),  # TODO: this should be adjusted from 1024
        'celeb256':    ModelSettings(beta_kl=1.0, beta_rec=0.5, beta_neg=1024, z_dim=256, batch_size=8),  # TODO: this should be adjusted from 1024
        'celeb1024':   ModelSettings(beta_kl=1.0, beta_rec=0.5, beta_neg=1024, z_dim=256, batch_size=8),
        'monsters128': ModelSettings(beta_kl=0.2, beta_rec=0.2, beta_neg=256,  z_dim=128, batch_size=16),
        'svhn':        ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_dim=128, batch_size=32),
        'fmnist':      ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_dim=32,  batch_size=128),
        'mnist':       ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_dim=32,  batch_size=128),
        'mtg':         ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=1024, z_dim=256, batch_size=32),  # verify settings
    }

    def main(dataset: str = 'mtg', wandb_enabled: bool = False):
        cfg = _SETTINGS[dataset]

        print('[initialising]: model')
        system = SoftIntroVaeSystem(
            dataset=dataset,
            beta_kl=cfg.beta_kl,
            beta_rec=cfg.beta_rec,
            beta_neg=cfg.beta_neg,
            z_dim=cfg.z_dim,
            lr_dec=2e-4,
            lr_enc=2e-4,
            lr_vae=2e-4,
            train_steps_vae=2000,
        )

        # initialise model
        # init_model_weights(system.model, mode='xavier_normal', verbose=True)

        # get dataset & visualise images
        print('[initialising]: data')
        mean_std = (0.5, 0.5)
        dataset  = DataLoader(system.dataset_settings.make_dataset(), batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count())
        vis_imgs = torch.stack([normalize_image_obs(dataset.dataset[i]) for i in range(5)])

        # start training model
        print('[initialising]: trainer')
        trainer = make_mtg_trainer(
            train_epochs=400,
            visualize_period=500,
            visualize_input={'recons': (vis_imgs, mean_std)},
            # wandb settings
            wandb_project='soft-intro-vae',
            wandb_name=f'{dataset}:{cfg.z_dim}',
            wandb_enabled=wandb_enabled,
        )

        trainer.fit(system, dataset)

    # ENTRYPOINT
    logging.basicConfig(level=logging.INFO)
    main(
        dataset='mnist',
        wandb_enabled=False,
    )
