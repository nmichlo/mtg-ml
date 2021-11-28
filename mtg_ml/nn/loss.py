
import numpy as np
import torch
from disent.nn.functional import get_kernel_size
from disent.nn.functional import torch_conv2d_channel_wise
from disent.nn.functional import torch_conv2d_channel_wise_fft
from torch import nn
from torch.distributions import Distribution
from torch.nn import functional as F


# ========================================================================= #
# Losses                                                                    #
# ========================================================================= #


def torch_laplace_kernel2d(size, sigma=1.0, normalise=True):
    # make kernel
    pos = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y = torch.meshgrid(pos, pos)
    # compute values
    norm = (x**2 + y**2) / (2 * sigma ** 2)
    # compute kernel
    kernel = - (2 - 2 * norm) * torch.exp(-norm) / (2 * np.pi * sigma ** 2)
    if normalise:
        kernel -= kernel.mean()
        # kernel /= torch.abs(kernel).sum()
    # return kernel
    return kernel[None, None, :, :]


class BceLogitsLoss(nn.Module):
    def forward(self, x, target, reduction='mean'):
        return F.binary_cross_entropy_with_logits(x, target, reduction=reduction)


class BceLoss(nn.Module):
    def forward(self, x, target, reduction='mean'):
        return F.binary_cross_entropy(x, target, reduction=reduction)


class MseLoss(nn.Module):
    def forward(self, x, target, reduction='mean'):
        return F.mse_loss(x, target, reduction=reduction)


class SpatialFreqLoss(nn.Module):
    """
    Modified from:
    https://arxiv.org/pdf/1806.02336.pdf
    """

    def __init__(self, sigmas=(0.8, 1.6, 3.2), truncate=3, fft=True):
        super().__init__()
        assert len(sigmas) > 0
        self._kernels = nn.ParameterList([
            nn.Parameter(torch_laplace_kernel2d(get_kernel_size(sigma=sigma, truncate=truncate), sigma=sigma, normalise=True), requires_grad=False)
            for sigma in sigmas
        ])
        print([k.shape for k in self._kernels])
        self._n = len(self._kernels)
        self._conv_fn = torch_conv2d_channel_wise_fft if fft else torch_conv2d_channel_wise

    def forward(self, x, target, reduction='mean'):
        loss_orig = F.mse_loss(x, target, reduction=reduction)
        loss_freq = 0
        for kernel in self._kernels:
            loss_freq += F.mse_loss(self._conv_fn(x, kernel), self._conv_fn(target, kernel), reduction=reduction)
        return (loss_orig + loss_freq) / (self._n + 1)


class LaplaceMseLoss(nn.Module):
    """
    Modified from:
    https://arxiv.org/pdf/1806.02336.pdf
    """

    def __init__(self, freq_ratio=0.5, lazy_init=False):
        super().__init__()
        self._ratio = freq_ratio
        self._kernel = None
        # if we want to switch the loss during training, or on an existing model, these params wont exist!
        if not lazy_init:
            self._init_kernel()

    def _init_kernel(self, device=None):
        if self._kernel is None:
            self._kernel = nn.Parameter(torch.as_tensor([
                [0,  1,  0],
                [1, -4,  1],
                [0,  1,  0],
            ], dtype=torch.float32, device=device), requires_grad=False)
            self.register_parameter('_kernel', self._kernel)

    def forward(self, x, target, reduction='mean'):
        # create
        self._init_kernel(x.device)
        # convolve
        x_conv = torch_conv2d_channel_wise(x, self._kernel)
        t_conv = torch_conv2d_channel_wise(target, self._kernel)
        loss_orig = F.mse_loss(x, target, reduction=reduction)
        loss_freq = F.mse_loss(x_conv, t_conv, reduction=reduction)
        return (1 - self._ratio) * loss_orig + self._ratio * loss_freq


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


def get_recon_loss(loss: str):
    # loss function
    if loss == 'mse':                return MseLoss()
    elif loss == 'bce':              return BceLoss()
    elif loss == 'bce_logits':       return BceLogitsLoss()
    elif loss == 'mse_laplace_0.25': return LaplaceMseLoss(freq_ratio=0.25)
    elif loss == 'mse_laplace_0.5':  return LaplaceMseLoss()
    elif loss == 'mse_spatial':      return SpatialFreqLoss()
    else:                            raise KeyError(f'invalid reconstruction loss: {repr(loss)}')


def kl_div(posterior: Distribution, prior: Distribution):
    # This is how the original VAE/BetaVAE papers do it.
    # - we compute the reverse kl divergence directly instead of approximating it
    # - kl(post|prior)
    # FORWARD vs. REVERSE kl (https://www.tuananhle.co.uk/notes/reverse-forward-kl.html)
    # - If we minimize the kl(post|prior) or the reverse/exclusive KL, the zero-forcing/mode-seeking behavior arises.
    # - If we minimize the kl(prior|post) or the forward/inclusive KL, the mass-covering/mean-seeking behavior arises.
    return torch.distributions.kl_divergence(posterior, prior)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
