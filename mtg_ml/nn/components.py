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
from typing import Union

import torch
from disent.nn.activations import Swish
from torch import nn as nn
from torch.distributions import Normal


# ========================================================================= #
# Activations                                                               #
# ========================================================================= #


def Activation(shape_or_features: Optional[Union[Sequence[int], int]] = None, activation: Optional[str] = 'leaky_relu', norm: Optional[str] = 'instance'):
    layers = []
    # make norm layer
    if (norm is not None) and (shape_or_features is not None):
        # get components
        if isinstance(shape_or_features, int):
            C, H, W = shape_or_features, None, None
        else:
            C, H, W = shape_or_features
        # get norm layers
        if norm == 'instance':    layers.append(nn.InstanceNorm2d(num_features=C, momentum=0.05))
        elif norm == 'batch':     layers.append(nn.BatchNorm2d(num_features=C,    momentum=0.05))
        elif norm == 'layer_hw':  layers.append(nn.LayerNorm(normalized_shape=[H, W]))
        elif norm == 'layer_chw': layers.append(nn.LayerNorm(normalized_shape=[C, H, W]))
        else:                     raise KeyError(f'invalid norm mode: {norm}')
    # make activation
    if activation is not None:
        if activation == 'swish':        layers.append(Swish())
        elif activation == 'leaky_relu': layers.append(nn.LeakyReLU(inplace=True))
        elif activation == 'relu':       layers.append(nn.ReLU(inplace=True))
        elif activation == 'relu6':      layers.append(nn.ReLU6(inplace=True))
        else:                            raise KeyError(f'invalid activation mode: {activation}')
    # return model
    if layers: return nn.Sequential(*layers)
    else:      return nn.Identity()


# ========================================================================= #
# DISTS                                                                     #
# ========================================================================= #


class NormalDist(nn.Module):
    def forward(self, z):
        assert z.ndim == 2, f'latent dimension was not flattened, could not instantiate posterior distribution with shape: {z.shape}'
        mu, log_var = z.chunk(2, dim=1)
        return Normal(loc=mu, scale=torch.exp(0.5 * log_var))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
