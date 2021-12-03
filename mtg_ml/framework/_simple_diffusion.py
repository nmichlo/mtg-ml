from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from disent.util.math.random import random_choice_prng
from torch.optim import RAdam

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
        latent_chn: int = 3,
        device='cpu',
    ):
        self._buffer_size = buffer_size
        self._img_size = img_size
        # default device
        self._device = device
        # filled values
        self._filled_mask = torch.zeros(buffer_size, device=self._device, dtype=torch.bool)
        self._iters       = torch.zeros(buffer_size, device=self._device, dtype=torch.int32)
        # buffers
        self._inputs = torch.zeros(buffer_size, 3 + latent_chn, img_size, img_size, device=self._device, dtype=torch.float32)
        self._targets = torch.zeros(buffer_size, 3, img_size, img_size, device=self._device, dtype=torch.float32)

    # def to(self, device):
    #     self._filled_mask = self._filled_mask.to(device=device)
    #     self._iters = self._iters.to(device=device)
    #     self._inputs = self._inputs.to(device=device)
    #     self._targets = self._targets.to(device=device)

    def sample_idxs(self, n: int):
        idxs = random_choice_prng(self._buffer_size, size=n, replace=False)
        return torch.from_numpy(idxs).to(device=self._device)

    def cat_valid_samples(self, n_extra: int, new_inputs: torch.Tensor, new_targets: torch.Tensor):
        n_new = len(new_inputs)
        # checks
        assert len(new_inputs) == len(new_targets)
        assert new_inputs.shape[1:] == self._inputs.shape[1:]
        assert new_targets.shape[1:] == self._targets.shape[1:]
        # sample idxs & split
        _idxs = self.sample_idxs(n_new + n_extra)
        new_idxs, extra_idxs   = _idxs[:n_new], _idxs[n_new:]
        # get valid extras
        extra_idxs = extra_idxs[self._filled_mask[extra_idxs]]
        extra_iters = self._iters[extra_idxs]
        extra_inputs = self._inputs[extra_idxs]
        extra_targets = self._targets[extra_idxs]
        # concatenate
        idxs = torch.cat([new_idxs, extra_idxs], dim=0)
        iters = torch.cat([torch.zeros(n_new, dtype=torch.int32, device=self._device), extra_iters],   dim=0)
        inputs = torch.cat([new_inputs, extra_inputs],  dim=0)
        targets = torch.cat([new_targets, extra_targets], dim=0)
        # done!
        return idxs, iters, inputs, targets

    def overwrite(self, idxs, iters, inputs: torch.Tensor, targets: torch.Tensor):
        self._iters[idxs] = iters.detach().to(device=self._device)
        self._inputs[idxs] = inputs.detach().to(device=self._device)
        self._targets[idxs] = targets.detach().to(device=self._device)
        self._filled_mask[idxs] = True


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
            latent_chn: int = 3,
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
            in_channels=3 + self.hparams.latent_chn,
            model_channels=self.hparams.model_channels,
            out_channels=3 + self.hparams.latent_chn,
            num_res_blocks=self.hparams.model_res_blocks,
            attention_resolutions=tuple(self.hparams.img_size // int(res) for res in self.hparams.model_attention_resolutions),
            dropout=0.0,
            channel_mult=tuple(self.hparams.model_channel_multipliers),
            use_checkpoint=False,
            num_heads=self.hparams.model_heads,
            num_heads_upsample=-1,
        )
        # make a buffer
        self._buffer: Buffer = None

    def configure_optimizers(self):
        # return AdamW(self._model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return RAdam(self._model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def extract_rgb(self, outputs: torch.Tensor):
        assert outputs.ndim == 4
        assert outputs.shape[1] == 3 + self.hparams.latent_chn
        return outputs[:, :3, :, :]

    def extract_latents(self, outputs: torch.Tensor):
        assert outputs.ndim == 4
        assert outputs.shape[1] == 3 + self.hparams.latent_chn
        return outputs[:, 3:, :, :]

    def pad_rgb(self, targets: torch.Tensor):
        assert targets.ndim == 4
        assert targets.shape[1] == 3
        n, c, h, w = targets.shape
        return torch.cat([targets, torch.zeros(n, self.hparams.latent_chn, h, w, dtype=torch.float32, device=targets.device)], dim=1)

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

    def on_train_start(self) -> None:
        self._buffer = Buffer(
            buffer_size=self.hparams.buffer_size,
            img_size=self.hparams.img_size,
            latent_chn=self.hparams.latent_chn,
            device=self.device
        )

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        # make starting organisms
        inputs = self._make_training_inputs(batch)
        # sample from buffer
        idxs, iters, inputs, targets = self._buffer.cat_valid_samples(self.hparams.buffer_samples, inputs, batch)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # feed forward
        for i in range(self.hparams.grad_steps):
            inputs = self._model(inputs)
        # update iters
        iters = iters + self.hparams.grad_steps
        print(iters)
        loss_scales = torch.clip(iters.to(torch.float32) / self.hparams.denoise_steps, 0, 1)
        # compute the loss
        loss = F.mse_loss(self.extract_rgb(inputs), targets, reduction='none').mean(dim=(-3, -2, -1))
        loss = (loss * loss_scales).mean()
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # update the buffer
        self._buffer.overwrite(idxs, iters, inputs, targets)
        # log values
        self.log('s', len(idxs) - len(batch), prog_bar=True)
        self.log('it', float(loss_scales.mean()), prog_bar=True)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # done!
        return loss

    def on_train_end(self):
        self._buffer = None

    def forward(self, x, steps: int = None, pad_rgb: Optional[bool] = None, extract_rgb: bool = True):
        assert x.ndim == 4
        assert x.shape[1] in (3, 3 + self.hparams.latent_chn)
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
            # 'zerod_zs': self.extract_latents(rs_zerod.detach()).cpu(),
            #
            'noise_xs': self.extract_rgb(xs_noise.detach()).cpu(),
            'noise_rs': self.extract_rgb(rs_noise.detach()).cpu(),
            # 'noise_zs': self.extract_latents(rs_noise.detach()).cpu(),
        }
