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
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution
from torch.distributions import Normal

from mtg_ml.nn.components import ActAndNorm
from mtg_ml.nn.components import NormalDistLayer
from mtg_ml.nn.components import TanhNormal
from mtg_ml.nn.components import TanhNormalDistLayer


# ========================================================================= #
# ENC Modules                                                               #
# ========================================================================= #


def ConvDown(in_channels: int, out_channels: int, kernel_size: int = 4, last_activation: bool = False):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        *([ActAndNorm(shape_or_features=out_channels)] if last_activation else []),
    )

# def ConvDown(in_channels: int, out_channels: int, kernel_size: int = 3):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
#             # Activation(shape_or_features=out_channels),
#         # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
#             nn.AvgPool2d(kernel_size=2),
#             Activation(shape_or_features=out_channels),
#     )

def ConvUp(in_channels: int, out_channels: int, kernel_size: int = 4, last_activation: bool = True):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        *([ActAndNorm(shape_or_features=None)] if last_activation else []),
    )

# def ConvUp(in_channels: int, out_channels: int, kernel_size: int = 3, last_activation=True):
#     return nn.Sequential(
#             nn.UpsamplingNearest2d(scale_factor=2),
#         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
#             # Activation(shape_or_features=out_channels),
#         # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
#             *([Activation(shape_or_features=out_channels)] if last_activation else []),
#     )


# ========================================================================= #
# Conv -> Repr                                                              #
# ========================================================================= #


def ReprDown(in_shape: Sequence[int], hidden_size: Optional[int], out_size: int):
    if hidden_size is None:
        return nn.Sequential(
                nn.Flatten(),
            nn.Linear(int(np.prod(in_shape)), out_size),
        )
    else:
        return nn.Sequential(
                nn.Flatten(),
            nn.Linear(int(np.prod(in_shape)), hidden_size),
                ActAndNorm(),
            nn.Linear(hidden_size, out_size),
        )


def ReprUp(in_size: int, hidden_size: Optional[int], out_shape: Sequence[int]):
    if hidden_size is None:
        return nn.Sequential(
            nn.Linear(in_size, int(np.prod(out_shape))),
                ActAndNorm(),
                nn.Unflatten(dim=1, unflattened_size=out_shape),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_size, hidden_size),
                ActAndNorm(),
            nn.Linear(hidden_size, int(np.prod(out_shape))),
                ActAndNorm(),
                nn.Unflatten(dim=1, unflattened_size=out_shape),
        )


# ========================================================================= #
# AE                                                                        #
# ========================================================================= #


class BaseGaussianVaeModel(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self._posterior_layer = self._make_posterior_layer()
        # this should take in a tensor of shape (B, C, H, W)
        # this should return a tensor of shape (B, 2 * LATENTS)
        # -> this is then wrapped with a normal distribution,
        #    the first half are the means, the second hald are
        #    the log(variances)
        self.encoder = encoder
        # this should take in a tensor of shape (B, LATENTS)
        # this should return a tensor of shape (B, C, H, W)
        self.decoder = decoder

    def _make_posterior_layer(self):
        return NormalDistLayer()

    def make_prior(self, posterior: Distribution) -> Distribution:
        assert isinstance(posterior, Normal)
        return Normal(
            loc=torch.zeros_like(posterior.loc, requires_grad=False),
            scale=torch.ones_like(posterior.scale, requires_grad=False),
        )

    def encode(self, x, return_prior=False) -> Union[Distribution, Tuple[Distribution, Distribution]]:
        posterior = self._posterior_layer(self.encoder(x))
        # return values
        if return_prior:
            return posterior, self.make_prior(posterior)
        return posterior

    def decode(self, z) -> torch.Tensor:
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x, detach_z=False) -> torch.Tensor:
        assert not self.training, 'model not in evaluation mode'
        # deterministic forward if evaluating
        posterior = self.encode(x)
        z = posterior.mean.detach() if detach_z else posterior.mean
        recon = self.decode(z)
        return recon

    def forward_train(self, x, detach_z=False) -> Tuple[torch.Tensor, torch.Tensor, Distribution, Distribution]:
        assert self.training, 'model not in training mode'
        # stochastic forward if training
        posterior, prior = self.encode(x, return_prior=True)
        z = posterior.rsample().detach() if detach_z else posterior.rsample()
        recon = self.decode(z)
        return recon, z, posterior, prior

    def sample_z(self, batch_size: int, z_size: int, numpy: bool = False, device=None):
        size = (batch_size, z_size)
        if numpy:
            return torch.from_numpy(np.random.randn(*size)).to(dtype=torch.float32, device=device)
        else:
            return torch.randn(size=size, dtype=torch.float32, device=device)


# ========================================================================= #
# Uniform VAE-like model                                                    #
# ========================================================================= #


class BaseTanhGaussianVaeModel(BaseGaussianVaeModel):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        posterior_scale: float = 1.0,
    ):
        super().__init__(encoder=encoder, decoder=decoder)
        self._posterior_scale = posterior_scale

    def _make_posterior_layer(self):
        return TanhNormalDistLayer()

    def make_prior(self, posterior: Distribution) -> Distribution:
        assert isinstance(posterior, TanhNormal)
        return TanhNormal(
            normal_loc=torch.zeros_like(posterior.base_dist.loc, requires_grad=False),
            normal_scale=torch.full_like(posterior.base_dist.scale, fill_value=self._posterior_scale, requires_grad=False),
        )

    @torch.no_grad()
    def sample_z(self, batch_size: int, z_size: int, numpy: bool = False, device=None):
        samples = super().sample_z(batch_size=batch_size, z_size=z_size, numpy=numpy, device=device)
        return torch.tanh(samples)


# ========================================================================= #
# AE                                                                        #
# ========================================================================= #



class SimpleGaussianVaeModel(BaseGaussianVaeModel):
    encoder = None
    decoder = None

    def __init__(self, z_size: int = 128, repr_channels: int = 16, repr_hidden_size: Optional[int] = None, channel_mul=1.5, channel_start=16):
        super().__init__()

        def c(i: int):
            return int(channel_start * (channel_mul**i))

        self._enc = nn.Sequential(
                ConvDown(in_channels=3,    out_channels=c(0)),
                ConvDown(in_channels=c(0), out_channels=c(1)),
                ConvDown(in_channels=c(1), out_channels=c(2)),
                ConvDown(in_channels=c(2), out_channels=c(3)),
                ConvDown(in_channels=c(3), out_channels=repr_channels),
            ReprDown(in_shape=[repr_channels, 7, 5], hidden_size=repr_hidden_size, out_size=z_size * 2),
        )

        self._dec = nn.Sequential(
            ReprUp(in_size=z_size, hidden_size=repr_hidden_size, out_shape=[repr_channels, 7, 5]),
                ConvUp(in_channels=repr_channels, out_channels=c(3)),
                ConvUp(in_channels=c(3), out_channels=c(2)),
                ConvUp(in_channels=c(2), out_channels=c(1)),
                ConvUp(in_channels=c(1), out_channels=c(0)),
                ConvUp(in_channels=c(0), out_channels=3, last_activation=False),
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
