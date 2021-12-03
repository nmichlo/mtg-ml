from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from disent.util.math.random import random_choice_prng
from torch.optim import AdamW

from mtg_ml.framework._improved_diffusion._util.unet_basic import UNetAttentionModel
from mtg_ml.util.ptl import MlSystem


# ========================================================================= #
# Buffer                                                                    #
# ========================================================================= #


class Buffer(object):

    def __init__(
        self,
        buffer_size: int = 1024*4,
        img_size: int = 64,
    ):
        self._buffer_size = buffer_size
        self._img_size = img_size
        # filled values
        self._filled_mask = torch.zeros(buffer_size, device='cpu', dtype=torch.bool)
        # buffers
        self._inputs = torch.zeros(buffer_size, 6, img_size, img_size, device='cpu', dtype=torch.float32)
        self._targets = torch.zeros(buffer_size, 3, img_size, img_size, device='cpu', dtype=torch.float32)

    def sample_idxs(self, n: int):
        return random_choice_prng(self._buffer_size, size=n, replace=False)

    def _get_valid(self, idxs, device=None):
        # only get indices that are filled
        idxs = idxs[self._filled_mask[idxs]]
        # get values
        inputs = self._inputs[idxs]
        targets = self._targets[idxs]
        # move to device
        if device is not None:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
        # done!
        return idxs, inputs, targets

    def cat_valid_samples(self, n_extra: int, inputs: torch.Tensor, targets: torch.Tensor, device=None):
        # checks
        assert len(inputs) == len(targets)
        assert inputs.shape[1:] == self._inputs.shape[1:]
        assert targets.shape[1:] == self._targets.shape[1:]
        # sample idxs
        n = len(inputs)
        idxs = self.sample_idxs(len(inputs) + n_extra)
        # get missing
        extra_idxs = idxs[n:]
        valid_idxs, valid_inputs, valid_targets = self._get_valid(extra_idxs, device=device)
        # concatenate
        cat_inputs = torch.cat([inputs, valid_inputs], dim=0)
        cat_targets = torch.cat([targets, valid_targets], dim=0)
        cat_idxs = np.concatenate([idxs[:n], valid_idxs], axis=0)
        # done!
        return cat_idxs, cat_inputs, cat_targets

    def overwrite(self, idxs, inputs: torch.Tensor, targets: torch.Tensor):
        self._inputs[idxs] = inputs.detach().cpu()
        self._targets[idxs] = targets.detach().cpu()
        self._filled_mask[idxs] = True

    # def store(self, inputs: torch.Tensor, targets: torch.Tensor):
    #     # check sizes
    #     assert len(inputs) == len(targets)
    #     assert inputs.shape[1:] == self._inputs[1:]
    #     assert targets.shape[1:] == self._targets[1:]
    #     # move to cpu
    #     inputs = inputs.detach().cpu()
    #     targets = targets.detach().cpu()
    #     # store everything
    #     if self._filled < self._buffer_size:
    #         n = min(len(inputs), self._buffer_size - self._filled)
    #         # insert elems
    #         self._inputs[self._filled:self._filled+n] = inputs
    #         self._targets[self._filled:self._filled+n] = targets
    #         # update filled
    #         self._filled_mask[self._filled:self._filled+n] = True
    #         self._filled += n
    #         # If batch is not a multiple of the buffer size, then we miss elements here.
    #         # This probably wont affect anything though.
    #         return
    #     # randomly replace
    #     idxs = random_choice_prng(self._buffer_size, size=len(inputs), replace=False)
    #     self._inputs[idxs] = inputs
    #     self._targets[idxs] = targets

    # def try_sample(self, n: int, device=None):
    #     # only get indices that are filled
    #     return self.get_valid(self.sample_idxs(n), device=device)


# ========================================================================= #
# System                                                                    #
# ========================================================================= #


class BasicDenoiser(MlSystem):

    def __init__(
        self, *,
        # data
            img_size: int = 64,
        # model
            model_channels: int = 32,
            model_res_blocks: int = 2,
            model_heads: int = 2,
            model_channel_multipliers: Sequence[int] = (1, 2, 3, 4),
            model_attention_resolutions: Sequence[int] = (16, 8),
        # optimizer
            lr: float = 1e-4,
            weight_decay: float = 0.0,
        # sampling & training
            denoise_steps: int = 100,
            grad_steps: int = 2,
        # buffer
            buffer_size: int = 1024*4,
            buffer_samples: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        # instantiate configs
        self._model = UNetAttentionModel(
            in_channels=6,
            model_channels=self.hparams.model_channels,
            out_channels=6,
            num_res_blocks=self.hparams.model_res_blocks,
            attention_resolutions=tuple(self.hparams.img_size // int(res) for res in self.hparams.model_attention_resolutions),
            dropout=0.0,
            channel_mult=tuple(self.hparams.model_channel_multipliers),
            use_checkpoint=False,
            num_heads=self.hparams.model_heads,
            num_heads_upsample=-1,
        )
        # make a buffer
        self._buffer = Buffer(
            buffer_size=self.hparams.buffer_size,
            img_size=self.hparams.img_size,
        )

    def configure_optimizers(self):
        return AdamW(self._model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def extract_rgb(self, outputs: torch.Tensor):
        assert outputs.ndim == 4
        assert outputs.shape[1] == 6
        return outputs[:, :3, :, :]

    def extract_latents(self, outputs: torch.Tensor):
        assert outputs.ndim == 4
        assert outputs.shape[1] == 6
        return outputs[:, 3:, :, :]

    def pad_rgb(self, targets: torch.Tensor):
        assert targets.ndim == 4
        assert targets.shape[1] == 3
        return torch.cat([targets, torch.zeros_like(targets, device=targets.device)], dim=1)

    def _make_training_inputs(self, targets: torch.Tensor):
        assert targets.ndim == 4
        assert targets.shape[1] == 3
        # create inputs
        noise       = torch.randn_like(targets, device=targets.device)
        noise_scale = torch.rand(len(targets), 1, 1, 1, device=targets.device)
        image_scale = torch.rand(len(targets), 1, 1, 1, device=targets.device)
        content = (targets * image_scale) + (noise * noise_scale)
        # done!
        return self.pad_rgb(content)

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        # make starting organisms
        inputs = self._make_training_inputs(batch)
        # sample from buffer
        idxs, inputs, targets = self._buffer.cat_valid_samples(self.hparams.buffer_samples, inputs, batch, device=self.device)
        # feed forward
        for i in range(self.hparams.grad_steps):
            inputs = self._model(inputs)
        # update the buffer
        self._buffer.overwrite(idxs, inputs, targets)
        # compute the loss
        loss = F.mse_loss(self.extract_rgb(inputs), targets, reduction='mean')
        # log values
        self.log('samp.', len(idxs) - self.hparams.buffer_samples, prog_bar=True)
        # done!
        return loss

    def forward(self, x, steps: int = None, pad_rgb: Optional[bool] = None, extract_rgb: bool = True):
        assert x.ndim == 4
        assert x.shape[1] in (3, 6)
        # get defaults
        if steps is None:
            steps = self.hparams.denoise_steps
        # pad
        if pad_rgb is None:
            pad_rgb = (x.shape[1] == 3)
        if pad_rgb:
            x = self.pad_rgb(x)
        # feed forward
        for i in range(steps):
            x = self._model(x)
        # unpad
        if extract_rgb:
            x = self.extract_rgb(x)
        # done!
        return x

    @torch.no_grad()
    def visualize_batch(self, xs: torch.Tensor) -> Dict[str, torch.Tensor]:
        xs = xs.to(self.device)
        # feed forward
        xs_zerod = self.pad_rgb(xs)
        xs_noise = self._make_training_inputs(xs)
        # examples
        rs_zerod = self(xs_zerod, extract_rgb=False)
        rs_noise = self(xs_noise, extract_rgb=False)
        # save
        return {
            'zerod_xs': self.extract_rgb(xs_zerod.detach()).cpu(),
            'zerod_rs': self.extract_rgb(rs_zerod.detach()).cpu(),
            'zerod_zs': self.extract_latents(rs_zerod.detach()).cpu(),
            #
            'noise_xs': self.extract_rgb(xs_noise.detach()).cpu(),
            'noise_rs': self.extract_rgb(rs_noise.detach()).cpu(),
            'noise_zs': self.extract_latents(rs_noise.detach()).cpu(),
        }
