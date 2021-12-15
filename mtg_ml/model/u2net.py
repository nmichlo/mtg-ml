"""
The code for our newly accepted paper in Pattern
Recognition 2020: "U^2-Net: Going Deeper with Nested
U-Structure for Salient Object Detection."

FROM: https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net_refactor.py

MODIFICATIONS:
- cleaned up for use the mtg_ml
- changed to recursive definition with multiple layers
- combined encoder & decoder configs into single stage configs to make modifications easier -- automatically compute needed input channel sizes!
- adjustable input and channel sizes
"""


from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtg_ml.util.pt import count_params


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def _upsample_to_shape(x: torch.Tensor, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


_ACTIVATIONS = {
    'sigmoid': torch.sigmoid,
    'relu': torch.relu,
    'selu': torch.selu,
    'softmax': torch.softmax,
    'tanh': torch.tanh,
}


def activate(x: torch.Tensor, activation: str):
    fn = _ACTIVATIONS[activation]
    return fn(x)


# ========================================================================= #
# Components                                                                #
# ========================================================================= #


class ConvBnReLU(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dilate: int = 1,
    ):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=dilate, dilation=dilate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ========================================================================= #
# RSU                                                                       #
# ========================================================================= #


class RsuEncDec(nn.Module):

    """
    MODEL:
    x ──> enc ─────────> + ──> dec ───> (main)
           │             │
         [down]         [up]
           │             │
           ╰─> [child] ──╯

    NOTES:
    - none of the layers should change the W & H of the tensors.
    """

    def __init__(self, enc_layer: nn.Module, dec_layer: nn.Module, child: Union['RsuEncDec', 'RsuMid'], resize_child: bool = True):
        super().__init__()
        self.resize_child = resize_child
        # modules
        self.enc_layer = enc_layer
        self.child = child
        self.dec_layer = dec_layer

    def forward(self, x: torch.Tensor):
        # encoder
        e = self.enc_layer(x)
        # downsample -> middle -> upsample
        if self.resize_child:
            m = F.max_pool2d(e, kernel_size=2, stride=2, ceil_mode=True)
            m = self.child(m)
            m = _upsample_to_shape(m, e.shape[2:])
        else:
            m = self.child(e)
        # decoder
        d = self.dec_layer(torch.cat((m, e), dim=1))
        # done!
        return d


class RsuMid(nn.Module):

    """
    MODEL:
    x ──> enc ──> (main)

    NOTES:
    - none of the layers should change the W & H of the tensors.
    """

    def __init__(self, mid_layer: nn.Module):
        super().__init__()
        self.mid_layer = mid_layer

    def forward(self, x: torch.Tensor):
        return self.mid_layer(x)


class Rsu(nn.Module):
    """
    UNET acting as a residual (?) layer

    x ──> conv ────── + ──> (out)
           │          │
           ╰─> UNET ──╯
    """

    def __init__(
        self,
        num_layers: int,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        dilated: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dilated = dilated
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # input channels
        self.inp = ConvBnReLU(in_ch=in_ch, out_ch=out_ch, dilate=1)
        # 1. middle stage is single conv
        # 2. stage directly around middle does not resize
        # 3. end stage adjusts channel sizes
        self.unet = RsuMid(mid_layer=ConvBnReLU(mid_ch, mid_ch, dilate=(2 ** (num_layers-1)) if dilated else 2))
        # create n-1 surrounding stages: i in [1, ..., num_layers-1]
        for l in reversed(range(1, num_layers)):
            inp = mid_ch if (l != 1) else out_ch
            out = mid_ch
            # add as child
            self.unet = RsuEncDec(
                enc_layer=ConvBnReLU(in_ch=inp,   out_ch=out, dilate=(2 ** (l-1)) if dilated else 1),
                dec_layer=ConvBnReLU(in_ch=out*2, out_ch=inp, dilate=(2 ** (l-1)) if dilated else 1),
                child=self.unet,
                resize_child=(not dilated) and (l != num_layers-1)
            )

    def forward(self, x):
        hxin = self.inp(x)
        hx1d = self.unet(hxin)
        return hx1d + hxin


# ========================================================================= #
# Unet Stages                                                               #
# ========================================================================= #


class EncDecStage(nn.Module):

    """
    MODEL:
    x ──> enc ─────────> + ──> dec ───> (out)
           │             │      │
          down           up    side ──> [extra_0] (list)
           │             │
           ╰─> [child] ──╯

    RECURSIVE MODEL:
    x ──> enc ───────────────────────────────> + ──> dec ───> (out)
           │                                   │      │
          down                                 up     side ─> [extra_0, extra_1]  (list)
           │   ╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮    │            ╭─────────────╯
           ╰─> ┊ enc ─────────> + ──> dec ┊ ───╯            │
               ┊  │             │      │  ╰┄┄┄┄┄┄┄┄┄┄┄┄┄╮   │
               ┊ down           up     side ────────────────╯
               ┊  │             │                       ┊
               ┊  ╰─> [child] ──╯                       ┊
               ╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯

    NOTES:
    - none of the layers should change the W & H of the tensors.
    """

    def __init__(self, enc_layer: Rsu, dec_layer: Rsu, side_layer: nn.Module, child: Union['EncDecStage', 'MidStage']):
        super().__init__()
        # modules
        self.enc_layer = enc_layer
        self.child = child
        self.dec_layer = dec_layer
        self.side_layer = side_layer

    def forward(self, x: torch.Tensor):
        # encoder
        e = self.enc_layer(x)
        # downsample -> middle -> upsample
        m = F.max_pool2d(e, kernel_size=2, stride=2, ceil_mode=True)
        D, S = self.child(m)
        m = D[0]
        m = _upsample_to_shape(m, e.shape[2:])
        # decoder
        d = self.dec_layer(torch.cat((m, e), dim=1))
        # output
        s = self.side_layer(d)
        # done!
        return [d] + D, [s] + S


class MidStage(nn.Module):

    """
    MODEL:
    x ──> enc ──> (main)
           │
          side ──> (extra)

    NOTES:
    - none of the layers should change the W & H of the tensors.
    """

    def __init__(self, mid_layer: Rsu, side_layer: nn.Module):
        super().__init__()
        self.mid_layer = mid_layer
        self.side_layer = side_layer

    def forward(self, x: torch.Tensor):
        # encoder & decoder
        d = self.mid_layer(x)
        # output
        s = self.side_layer(d)
        # done!
        return [d], [s]


# ========================================================================= #
# Models                                                                    #
# ========================================================================= #


@dataclass
class U2NetStageCfg:
    layers: int
    enc_mid_ch: int
    enc_out_ch: int
    dec_mid_ch: int
    dec_out_ch: int
    dilated: bool


@dataclass
class U2NetMidCfg:
    layers: int
    mid_ch: int
    out_ch: int
    dilated: bool


class U2Net(nn.Module):
    """
    Effectively a UNET of differing UNETs (RSU layers)
    acting at different scales.
    """

    def __init__(
        self,
        stage_cfgs: List[U2NetStageCfg],  # should be in order from parent to child stages
        mid_cfg: U2NetMidCfg,
        in_ch: int = 3,
        out_ch: int = 1,
        out_activation: Optional[str] = 'sigmoid',
    ):
        super().__init__()
        # save hparams
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.out_activation = out_activation
        self.stage_cfgs = stage_cfgs[::-1]  # [ends, ..., mid] -> [mid, ..., ends]
        self.mid_cfg = mid_cfg
        self.num_stages = len(self.stage_cfgs) + 1
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # precompute input channels:
        self.parent_enc_out_chs = tuple([cfg.enc_out_ch for cfg in self.stage_cfgs] + [self.in_ch])                 # number of channels passed to child (by the encoder)
        self.child_dec_out_chs  = tuple([None, self.mid_cfg.out_ch] + [cfg.dec_out_ch for cfg in self.stage_cfgs][:-1])  # number of channels output by stage (from the decoder) ... middle layer has no child, so None!
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # create stages
        self.unet = MidStage(
            mid_layer=self._make_rsu_layer(num_layers=self.mid_cfg.layers, in_ch=self.parent_enc_out_chs[0], mid_ch=self.mid_cfg.mid_ch, out_ch=self.mid_cfg.out_ch, dilated=mid_cfg.dilated),
            side_layer=self._make_side_layer(dec_out_ch=self.mid_cfg.out_ch),
        )
        for i, cfg in enumerate(self.stage_cfgs):
            self.unet = EncDecStage(
                enc_layer=self._make_rsu_layer(num_layers=cfg.layers, in_ch=self.parent_enc_out_chs[i+1],                 mid_ch=cfg.enc_mid_ch, out_ch=cfg.enc_out_ch, dilated=cfg.dilated),
                dec_layer=self._make_rsu_layer(num_layers=cfg.layers, in_ch=cfg.enc_out_ch + self.child_dec_out_chs[i+1], mid_ch=cfg.dec_mid_ch, out_ch=cfg.dec_out_ch, dilated=cfg.dilated),
                side_layer=self._make_side_layer(dec_out_ch=cfg.dec_out_ch),
                child=self.unet,
            )
        # output layer -- combine all the side outputs together into a single one...
        self.out = self._make_output_layer()
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    def forward(self, x):
        # feed forward through unet and get side outputs
        _, S = self.unet(x)
        # generate outputs:
        # | 1. rescale
        S = [_upsample_to_shape(s, size=x.shape[2:]) for s in S]
        # | 2. concatenate & activate | TODO: this feels odd, should probably concat outputs from decoders, not side outputs, but still train individual stages on side outputs
        S = [self.out(torch.cat(S, dim=1)), *S]
        # | 3. activate
        if self.out_activation is not None:
            S = [activate(s, self.out_activation) for s in S]
        # done!
        return S

    def _make_rsu_layer(self, num_layers: int, in_ch: int, mid_ch: int, out_ch: int, dilated: bool):
        return Rsu(num_layers=num_layers, in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch, dilated=dilated)

    def _make_side_layer(self, dec_out_ch: int):
        return nn.Conv2d(in_channels=dec_out_ch, out_channels=self.out_ch, kernel_size=3, padding=1)  # TODO: a single conv layer feels odd, should maybe have an additional layer?

    def _make_output_layer(self):
        return nn.Conv2d(in_channels=self.num_stages * self.out_ch, out_channels=self.out_ch, kernel_size=1)  # TODO: a single conv layer feels odd, should maybe have a larger receptive field and an additional layer?


# ========================================================================= #
# U2Net Factory                                                             #
# ========================================================================= #


def make_u2net_full(in_ch: int = 3, out_ch: int = 1, out_activation: Optional[str] = 'sigmoid', cls=U2Net, **cls_kwargs):
    # stage configs
    stage_cfgs = [
        U2NetStageCfg(layers=7, enc_mid_ch=32,  enc_out_ch=64,  dec_mid_ch=16,  dec_out_ch=64,  dilated=False),
        U2NetStageCfg(layers=6, enc_mid_ch=32,  enc_out_ch=128, dec_mid_ch=32,  dec_out_ch=64,  dilated=False),
        U2NetStageCfg(layers=5, enc_mid_ch=64,  enc_out_ch=256, dec_mid_ch=64,  dec_out_ch=128, dilated=False),
        U2NetStageCfg(layers=4, enc_mid_ch=128, enc_out_ch=512, dec_mid_ch=128, dec_out_ch=256, dilated=False),
        U2NetStageCfg(layers=4, enc_mid_ch=256, enc_out_ch=512, dec_mid_ch=256, dec_out_ch=512, dilated=True),
    ]
    mid_cfg = U2NetMidCfg(layers=4, mid_ch=256, out_ch=512, dilated=True)
    # make network
    return cls(stage_cfgs=stage_cfgs, mid_cfg=mid_cfg, in_ch=in_ch, out_ch=out_ch, out_activation=out_activation, **cls_kwargs)


def make_u2net_lite(in_ch: int = 3, out_ch: int = 1, out_activation: Optional[str] = 'sigmoid', cls=U2Net, **cls_kwargs):
    # stage configs
    stage_cfgs = [
        U2NetStageCfg(layers=7, enc_mid_ch=16, enc_out_ch=64, dec_mid_ch=16, dec_out_ch=64, dilated=False),
        U2NetStageCfg(layers=6, enc_mid_ch=16, enc_out_ch=64, dec_mid_ch=16, dec_out_ch=64, dilated=False),
        U2NetStageCfg(layers=5, enc_mid_ch=16, enc_out_ch=64, dec_mid_ch=16, dec_out_ch=64, dilated=False),
        U2NetStageCfg(layers=4, enc_mid_ch=16, enc_out_ch=64, dec_mid_ch=16, dec_out_ch=64, dilated=False),
        U2NetStageCfg(layers=4, enc_mid_ch=16, enc_out_ch=64, dec_mid_ch=16, dec_out_ch=64, dilated=True),
    ]
    mid_cfg = U2NetMidCfg(layers=4, mid_ch=16, out_ch=64, dilated=True)
    # make network
    return cls(stage_cfgs=stage_cfgs, mid_cfg=mid_cfg, in_ch=in_ch, out_ch=out_ch, out_activation=out_activation, **cls_kwargs)


# ========================================================================= #
# Custom U2Net                                                              #
# ========================================================================= #


class U2NetAlt(U2Net):

    OUTPUT_LAYER_TYPES = ('default', 'conv1', 'conv3+conv1')

    def __init__(
        self,
        stage_cfgs: List[U2NetStageCfg],
        mid_cfg: U2NetMidCfg,
        in_ch: int,
        out_ch: int,
        out_activation: str = 'sigmoid',
        # custom
        out_layer_type: str = 'default',
        out_layers_ch: Optional[int] = None,
    ):
        if out_layer_type not in self.OUTPUT_LAYER_TYPES:
            raise KeyError(f'invalid out_layer_type={repr(out_layer_type)}, must be one of: {sorted(self.OUTPUT_LAYER_TYPES)}')
        self.out_layer_type = out_layer_type
        self.out_layers_ch = out_layers_ch
        # initialise
        super().__init__(stage_cfgs=stage_cfgs, mid_cfg=mid_cfg, in_ch=in_ch, out_ch=out_ch, out_activation=out_activation)
        # make output layers -- this should be similar to side channels
        if self.out_layers_ch is not None:
            self.out_layers = nn.ModuleList([self._make_pre_out_layer(dec_out_ch=dec_out_ch) for dec_out_ch in self.child_dec_out_chs[1:]])

    def _make_pre_out_layer(self, dec_out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_channels=dec_out_ch, out_channels=self.out_layers_ch, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

    def _make_output_layer(self):
        # get the number of input channels
        in_ch = self.num_stages * (self.out_ch if (self.out_layers_ch is None) else self.out_layers_ch)
        # get the type of layer
        if self.out_layer_type in ['conv3+conv1']:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_ch, out_channels=self.out_ch, kernel_size=1),
            )
        elif self.out_layer_type in ['default', 'conv1']:
            return nn.Conv2d(in_channels=in_ch, out_channels=self.out_ch, kernel_size=1)
        else:
            raise RuntimeError('this should never happen!')

    def forward(self, x):
        # feed forward through unet and get side outputs
        D, S = self.unet(x)
        # generate outputs:
        # | 1. rescale
        S = [_upsample_to_shape(s, size=x.shape[2:]) for s in S]
        # | 2.a activate
        if self.out_layers_ch is None:
            s = self.out(torch.cat(S, dim=1))
        else:
            D = [_upsample_to_shape(out_layer(d), size=x.shape[2:]) for d, out_layer in zip(D, self.out_layers)]
            s = self.out(torch.cat(D, dim=1))
        # | 2.b concatenate
        S = [s, *S]
        # | 3. activate
        if self.out_activation is not None:
            S = [activate(s, self.out_activation) for s in S]
        # done!
        return S


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    x = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    model = make_u2net_lite()
    print(count_params(model))
    model = make_u2net_lite(cls=U2NetAlt, out_layer_type='conv1',       out_layers_ch=None)
    print(count_params(model))
    model = make_u2net_lite(cls=U2NetAlt, out_layer_type='conv3+conv1', out_layers_ch=None)
    print(count_params(model))
    model = make_u2net_lite(cls=U2NetAlt, out_layer_type='conv1',       out_layers_ch=8)
    print(count_params(model))
    model = make_u2net_lite(cls=U2NetAlt, out_layer_type='conv3+conv1', out_layers_ch=8)
    print(count_params(model))

    outs = model(x)
    print([float(o.mean()) for o in outs])
    print([float(o.std()) for o in outs])
    print([tuple(o.shape) for o in outs])
