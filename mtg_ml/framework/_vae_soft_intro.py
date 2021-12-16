"""
Converted to pytorch_lightning from:
https://github.com/taldatech/soft-intro-vae-pytorch
"""

import dataclasses
from typing import Dict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from mtg_ml import DATASET_ROOT
from mtg_ml.nn.components import Act
from mtg_ml.nn.model import BaseGaussianVaeModel
from mtg_ml.nn.loss import get_recon_loss
from mtg_ml.nn.loss import kl_div

from mtg_ml.util.ptl import MlSystem
from mtg_ml.util.transforms import get_image_batch


# ========================================================================= #
# Model                                                                     #
# ========================================================================= #


class ResidualBlock(nn.Module):
    """
    ORIG: https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)

    FROM: https://github.com/taldatech/soft-intro-vae-pytorch
    """

    def __init__(
        self,
        inc: int = 64,
        outc: int = 64,
        groups: int = 1,
        scale: float = 1.0,
        activation: str = 'leaky_relu_0.2',
    ):
        super().__init__()

        midc = int(outc * scale)

        if inc != outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
            self.conv_expand = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1   = nn.BatchNorm2d(midc)
        self.relu1 = Act(activation=activation)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2   = nn.BatchNorm2d(outc)
        self.relu2 = Act(activation=activation)

    def forward(self, x):
        skip = self.conv_expand(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)) + skip)
        return x


# ========================================================================= #
# Encoder Decoder                                                           #
# ========================================================================= #


class SoftIntroEncoder(nn.Module):
    """
    FROM: https://github.com/taldatech/soft-intro-vae-pytorch
    """

    def __init__(
        self,
        img_size: int = 256,
        img_chn: int = 3,
        z_size: int = 512,
        double_fc: bool = False,
        conv_channels=(64, 128, 256, 512, 512, 512),
        activation: str = 'leaky_relu_0.2',
    ):
        super().__init__()

        self.img_chn = img_chn
        self.img_size = img_size
        cc = conv_channels[0]

        self.main = nn.Sequential(
            nn.Conv2d(img_chn, cc, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(cc),
            Act(activation=activation),
            nn.AvgPool2d(2),
        )

        sz = img_size // 2
        for ch in conv_channels[1:]:
            self.main.add_module(f'res_in_{sz}',     ResidualBlock(cc, ch, scale=1.0, activation=activation))
            self.main.add_module(f'down_to_{sz//2}', nn.AvgPool2d(2))
            cc, sz = ch, sz // 2
        self.main.add_module(f'res_in_{sz}', ResidualBlock(cc, cc, scale=1.0, activation=activation))

        # compute input shape to fcn | feed forward data!
        self.conv_output_size = self.main(torch.zeros(1, img_chn, img_size, img_size)).shape[1:]
        num_fc_features       = int(np.prod(self.conv_output_size))
        print(f'ENC: conv_output_size={self.conv_output_size} [{num_fc_features}]')

        # final layer
        if double_fc:
            self.fc = nn.Sequential(
                nn.Linear(num_fc_features, int(1.5 * z_size)),
                Act(activation=activation),
                nn.Linear(int(1.5 * z_size), 2 * z_size),
            )
        else:
            self.fc = nn.Linear(num_fc_features, 2 * z_size)

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        return y


class SoftIntroDecoder(nn.Module):
    """
    FROM: https://github.com/taldatech/soft-intro-vae-pytorch
    """

    def __init__(
        self,
        conv_input_size,
        img_size: int = 256,
        img_chn: int = 3,
        z_size: int = 512,
        double_fc: bool = False,
        conv_channels=(64, 128, 256, 512, 512, 512),
        activation: str = 'leaky_relu_0.2'
    ):
        super().__init__()
        self.img_chn = img_chn
        self.img_size = img_size
        cc = conv_channels[-1]

        # TODO: I shouldn't have to pass this
        self.conv_input_size = conv_input_size
        num_fc_features = int(np.prod(conv_input_size))
        print(f'DEC: conv_input_size={self.conv_input_size} [{num_fc_features}]')

        if double_fc:
            self.fc = nn.Sequential(
                nn.Linear(z_size, int(1.5 * z_size)),
                Act(activation=activation),
                nn.Linear(int(1.5 * z_size), num_fc_features),
                Act(activation=activation),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(z_size, num_fc_features),
                Act(activation=activation),
            )

        sz = 4
        self.main = nn.Sequential()
        for ch in conv_channels[::-1]:
            self.main.add_module(f'res_in_{sz}',  ResidualBlock(cc, ch, scale=1.0, activation=activation))
            self.main.add_module(f'up_to_{sz*2}', nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module(f'res_in_{sz}', ResidualBlock(cc, cc, scale=1.0, activation=activation))
        self.main.add_module('predict', nn.Conv2d(cc, img_chn, kernel_size=5, stride=1, padding=2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


# ========================================================================= #
# Model                                                                     #
# ========================================================================= #


class SoftIntroVaeModel(BaseGaussianVaeModel):

    def __init__(
        self,
        img_size: int = 256,
        img_chn: int = 3,
        z_size: int = 512,
        conv_channels=(64, 128, 256, 512, 512, 512),
        double_fc: bool = False,
        activation: str = 'leaky_relu_0.2',
    ):
        super().__init__()
        self.z_size = z_size
        self._encoder = SoftIntroEncoder(img_size=img_size, img_chn=img_chn, z_size=z_size, double_fc=double_fc, activation=activation, conv_channels=conv_channels)
        self._decoder = SoftIntroDecoder(img_size=img_size, img_chn=img_chn, z_size=z_size, double_fc=double_fc, activation=activation, conv_channels=conv_channels, conv_input_size=self._encoder.conv_output_size)

    def _enc(self, x):
        return self._encoder(x)

    def _dec(self, z):
        return self._decoder(z)


# ========================================================================= #
# Framework                                                                 #
# ========================================================================= #


class SoftIntroVaeSystem(MlSystem):

    def __init__(
        self,
        # model
            z_size: int = 128,
            model_double_fc: bool = True,
            activation: str = 'leaky_relu_0.2',
        # training
            lr_enc: float = 2e-4,
            lr_dec: float = 2e-4,
            recon_loss: str = "mse",
        # loss scaling
            beta_kl: float = 1.0,
            beta_rec: float = 1.0,
            beta_neg: float = 256.0,
            gamma_r: float = 1e-8,
        # dataset_settings:
            img_size: int = 128,
            img_chn: int = 3,
            conv_channels: Tuple[int, ...] = (64, 128, 256, 512, 512),
        # unimplemented:
            # exit_on_negative_diff: bool = False,  # TODO: implement
            # training_measure_fid: bool = False,   # TODO: implement
    ):
        super().__init__()
        self.save_hyperparameters()
        # initialise
        self._scale = 1 / (self.hparams.img_chn * self.hparams.img_size**2)  # 1 / (C * H * W)
        self._loss = get_recon_loss(self.hparams.recon_loss)
        # make model -- TODO: this model is quite terrible...?
        self.model = SoftIntroVaeModel(
            img_size=self.hparams.img_size,
            img_chn=self.hparams.img_chn,
            z_size=self.hparams.z_size,
            conv_channels=self.hparams.conv_channels,
            double_fc=self.hparams.model_double_fc,
            activation=self.hparams.activation,
        )

    def configure_optimizers(self):
        # TODO: select optimizer
        optimizer_enc = optim.AdamW(self.model._encoder.parameters(), lr=self.hparams.lr_enc)  # was Adam
        optimizer_dec = optim.AdamW(self.model._decoder.parameters(), lr=self.hparams.lr_dec)  # was Adam
        # enc_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_enc, milestones=(350,), gamma=1.0)
        # dec_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_dec, milestones=(350,), gamma=1.0)
        # TODO: update the schedule!
        # enc_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_enc, milestones=[64, 96, 128, 160], gamma=1/(10**0.5))  # was: milestones=(350,), gamma=0.1
        # dec_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_dec, milestones=[64, 96, 128, 160], gamma=1/(10**0.5))  # was: milestones=(350,), gamma=0.1
        enc_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_enc, milestones=[64, 192, 320], gamma=1/(10**0.5))  # was: milestones=(350,), gamma=0.1
        dec_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_dec, milestones=[64, 192, 320], gamma=1/(10**0.5))  # was: milestones=(350,), gamma=0.1
        return [optimizer_enc, optimizer_dec], [enc_scheduler, dec_scheduler]

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        # always 4 dimensions & a single tensor
        batch = get_image_batch(batch)

        # ENCODER TRAINING STEP
        if optimizer_idx == 0:
            # DO WE REALLY NEED THIS?
            # for param in self.model._encoder.parameters(): param.requires_grad = True
            # for param in self.model._decoder.parameters(): param.requires_grad = False

            recon_fake = self.model.decode(self.sample_z(len(batch)))

            # feed forward
            rec_real, z_real, posterior_real, prior_real = self.model.forward_train(batch)
            rec_rec,  z_rec,  posterior_rec,  prior_rec  = self.model.forward_train(rec_real.detach())
            rec_fake, z_fake, posterior_fake, prior_fake = self.model.forward_train(recon_fake.detach())

            # reconstruction losses
            loss_real = torch.flatten(self._loss(rec_real, batch,      reduction="none"), start_dim=1).sum(dim=-1).mean(dim=0)
            loss_rec  = torch.flatten(self._loss(rec_rec,  rec_real,   reduction='none'), start_dim=1).sum(dim=-1)
            loss_fake = torch.flatten(self._loss(rec_fake, recon_fake, reduction='none'), start_dim=1).sum(dim=-1)

            # kl divergences
            kl_real = kl_div(posterior_real, prior_real).sum(dim=-1).mean()
            kl_rec  = kl_div(posterior_rec,  prior_rec) .sum(dim=-1)
            kl_fake = kl_div(posterior_fake, prior_fake).sum(dim=-1)

            # exp elbo
            expelbo_rec  = (-2 * self._scale * (self.hparams.beta_rec * loss_rec  + self.hparams.beta_neg * kl_rec)).exp().mean(dim=0)
            expelbo_fake = (-2 * self._scale * (self.hparams.beta_rec * loss_fake + self.hparams.beta_neg * kl_fake)).exp().mean(dim=0)

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

            # handle invalid KL -- TODO
            # diff_kls.append(-kl_real.item() + kl_fake.item())
            # if np.mean(diff_kls) < -1.0:
            #     print(f'the kl difference [{np.mean(diff_kls):.3f}] between fake and real is negative (no sampling improvement)')
            #     print("try to lower beta_neg hyperparameter")
            #     print("exiting...")
            #     raise RuntimeError("Negative KL Difference")

            # backprop (must only update encoder)
            return loss

        # DECODER TRAINING STEP
        elif optimizer_idx == 1:
            # DO WE REALLY NEED THIS?
            # for param in self.model._encoder.parameters(): param.requires_grad = False
            # for param in self.model._decoder.parameters(): param.requires_grad = True

            # this should originally be re-used from the previous step TODO: check which other values should also be reused?
            batch_fake = self.model.decode(self.sample_z(len(batch)))

            # feed forward
            rec_real, z_real, posterior_real, prior_real = self.model.forward_train(batch, detach_z=True)
            rec_rec,  z_rec,  posterior_rec,  prior_rec  = self.model.forward_train(rec_real, detach_z=True)
            rec_fake, z_fake, posterior_fake, prior_fake = self.model.forward_train(batch_fake, detach_z=True)

            # kl divergences
            loss_real = torch.flatten(self._loss(rec_real, batch,               reduction="none"), start_dim=1).sum(dim=-1).mean(dim=0)
            loss_rec  = torch.flatten(self._loss(rec_rec,  rec_real.detach(),   reduction="none"), start_dim=1).sum(dim=-1).mean(dim=0)
            loss_fake = torch.flatten(self._loss(rec_fake, batch_fake.detach(), reduction="none"), start_dim=1).sum(dim=-1).mean(dim=0)

            # kl divergences
            kl_rec  = kl_div(posterior_rec,  prior_rec).sum(dim=-1).mean()
            kl_fake = kl_div(posterior_fake, prior_fake).sum(dim=-1).mean()

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

        # make sure we handle all cases!
        else:
            raise RuntimeError('this should never happen!')

    def forward(self, x):
        return self.model(x)

    def sample_z(self, batch_size: int, numpy=False):
        size = (batch_size, self.hparams.z_size)
        if numpy:
            return torch.from_numpy(np.random.randn(*size)).to(dtype=torch.float32, device=self.device)
        else:
            return torch.randn(size=size, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def visualize_batch(self, xs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # feed forward
        xs = xs.to(self.device)
        xs_recons = self.model.forward(xs)

        # get random samples
        zs = self.sample_z(len(xs), numpy=True)
        zs_recons = self.model.decode(zs)

        # save
        return {
            'xs': xs.detach().cpu(),
            'xs_recons': xs_recons.detach().cpu(),
            'zs_recons': zs_recons.detach().cpu(),
        }


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #


def CelebA(train_size: int, download: bool = False, root=DATASET_ROOT, transform=None):
    from torchvision.datasets import CelebA as _CelebA, ImageFolder
    import os
    # verify celeba | manual download from: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
    data = _CelebA(root, download=download)
    # load images directly
    data = ImageFolder(os.path.join(data.root, data.base_folder), transform=transform)
    # original partition code: [x for x in os.listdir('data/celeba/img_align_celeba') if is_image_file(x)][:162770]
    data = torch.utils.data.Subset(data, indices=list(range(train_size)))
    return data


_MAKE_DATASET_FNS = {
    'celeb128':    lambda: CelebA(train_size=162770, download=True, transform=transforms.Compose([transforms.Resize(128), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
    'celeb256':    lambda: CelebA(train_size=162770, download=True, transform=transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
    'celeb1024':   lambda: CelebA(train_size=29000,  download=True, transform=transforms.Compose([transforms.Resize(1024), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
#   'monsters128': lambda: DigitalMonstersDataset(root_path=os.path.join(_DATA_ROOT, 'monsters'), output_height=128 , transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
    'svhn':        lambda:         torchvision.datasets.SVHN(root=DATASET_ROOT, split='train', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
    'cifar10':     lambda:     torchvision.datasets.CIFAR10(root=DATASET_ROOT,  train=True,    download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
    'fmnist':      lambda: torchvision.datasets.FashionMNIST(root=DATASET_ROOT, train=True,    download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
    'mnist':       lambda:        torchvision.datasets.MNIST(root=DATASET_ROOT, train=True,    download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])),
}


@dataclasses.dataclass
class DatasetSettings:
    img_size: int
    img_chn: int
    conv_channels: Tuple[int, ...]


# settings from https://github.com/taldatech/soft-intro-vae-pytorch
_DATASET_SETTINGS = {
    'cifar10':     DatasetSettings(img_size=32,   img_chn=3, conv_channels=(64, 128, 256)),
    'celeb128':    DatasetSettings(img_size=128,  img_chn=3, conv_channels=(64, 128, 256, 512, 512)),
    'celeb256':    DatasetSettings(img_size=256,  img_chn=3, conv_channels=(64, 128, 256, 512, 512, 512)),
    'celeb1024':   DatasetSettings(img_size=1024, img_chn=3, conv_channels=(16, 32, 64, 128, 256, 512, 512, 512)),
  # 'monsters128': DatasetSettings(img_size=128,  img_chn=3, conv_channels=(64, 128, 256, 512, 512),
    'svhn':        DatasetSettings(img_size=32,   img_chn=3, conv_channels=(64, 128, 256)),
    'fmnist':      DatasetSettings(img_size=28,   img_chn=1, conv_channels=(64, 128)),
    'mnist':       DatasetSettings(img_size=28,   img_chn=1, conv_channels=(64, 128)),
}


@dataclasses.dataclass
class ModelSettings:
    beta_kl: float
    beta_rec: float
    beta_neg: float
    z_size: int
    batch_size: int


_SETTINGS = {
    'cifar10':     ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_size=128, batch_size=32),
    'celeb128':    ModelSettings(beta_kl=1.0, beta_rec=0.5, beta_neg=1024, z_size=256, batch_size=8),  # TODO: this should be adjusted from 1024
    'celeb256':    ModelSettings(beta_kl=1.0, beta_rec=0.5, beta_neg=1024, z_size=256, batch_size=8),  # TODO: this should be adjusted from 1024
    'celeb1024':   ModelSettings(beta_kl=1.0, beta_rec=0.5, beta_neg=1024, z_size=256, batch_size=8),
    'monsters128': ModelSettings(beta_kl=0.2, beta_rec=0.2, beta_neg=256,  z_size=128, batch_size=16),
    'svhn':        ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_size=128, batch_size=32),
    'fmnist':      ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_size=32,  batch_size=128),
    'mnist':       ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=256,  z_size=32,  batch_size=128),
    'mtg':         ModelSettings(beta_kl=1.0, beta_rec=1.0, beta_neg=1024, z_size=256, batch_size=32),  # verify settings
}


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
