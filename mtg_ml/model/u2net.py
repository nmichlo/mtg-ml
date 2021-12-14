
"""
The code for our newly accepted paper in Pattern
Recognition 2020: "U^2-Net: Going Deeper with Nested
U-Structure for Salient Object Detection."

FROM: https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net_refactor.py

MODIFICATIONS:
- cleaned up for use the mtg_ml
-
"""


from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # encoder & decoder
        d = self.mid_layer(x)
        # done!
        return d


# class Rsu(nn.Module):
#     def __init__(
#         self,
#         num_layers: int,
#         in_ch: int,
#         mid_ch: int,
#         out_ch: int,
#         dilated: bool = False,
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.dilated = dilated
#         self._make_layers(num_layers, in_ch, mid_ch, out_ch, dilated)
#
#     @staticmethod
#     def _size_map(x: torch.Tensor, height: int, offset: bool = False):
#         sizes = {}
#         # recursive
#         size = list(x.shape[-2:])
#         for h in range(0, height - 1):
#             sizes[h] = size
#             size = [math.ceil(s / 2) for s in size]
#         # done
#         if offset:
#             return {h + 1: s for h, s in sizes.items()}  # old style
#         return sizes
#
#     def forward(self, x):
#         sizes = self._size_map(x, self.num_layers - 1, offset=False)
#         x = self.rebnconvin(x)
#
#         # U-Net like symmetric encoder-decoder structure
#         def forward_layer(x, layer_idx: int):
#             if layer_idx < self.num_layers - 1:
#                 e = getattr(self, f'rebnconv{layer_idx}')(x)
#
#                 if not self.dilated and layer_idx < self.num_layers - 2:
#                     m = self.downsample(e)
#                     m = forward_layer(m, layer_idx + 1)
#                 else:
#                     m = forward_layer(e, layer_idx + 1)
#
#                 d = getattr(self, f'rebnconv{layer_idx}d')(torch.cat((m, e), 1))
#
#                 return _upsample_to_shape(d, sizes[layer_idx - 1]) if (not self.dilated and layer_idx > 0) else d
#             else:
#                 return getattr(self, f'rebnconv{layer_idx}')(x)
#
#         return x + forward_layer(x, layer_idx=0)
#
#     def _make_layers(self, num_layers: int, in_ch: int, mid_ch: int, out_ch: int, dilated: bool = False):
#         self.rebnconvin = ConvBnReLU(in_ch, out_ch)
#         self.downsample = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv0 = ConvBnReLU(out_ch, mid_ch)
#         self.rebnconv0d = ConvBnReLU(mid_ch * 2, out_ch)
#
#         for i in range(1, num_layers):
#             dilate = 1 if (not dilated) else 2 ** i
#             self.add_module(f'rebnconv{i}', ConvBnReLU(mid_ch, mid_ch, dilate=dilate))
#             self.add_module(f'rebnconv{i}d', ConvBnReLU(mid_ch * 2, mid_ch, dilate=dilate))
#
#         dilate = 2 if (not dilated) else (2 ** num_layers)
#         self.add_module(f'rebnconv{num_layers}', ConvBnReLU(mid_ch, mid_ch, dilate=dilate))


class Rsu(nn.Module):
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
    x ──> enc ─────────> + ──> dec ───> (main)
           │             │      │
          down           up    side ──> (extra)
           │             │
           ╰─> [child] ──╯

    RECURSIVE MODEL:
    x ──> enc ───────────────────────────────> + ──> dec ───> (main)
           │                                   │      │
          down                                 up     side ──> (extra)
           │   ╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮    │
           ╰─> ┊ enc ─────────> + ──> dec ┊ ───╯
               ┊  │             │      │  ╰┄┄┄┄┄┄┄┄┄┄┄┄┄╮
               ┊ down           up     side ──> (extra) ┊
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
        m, S = self.child(m)
        m = _upsample_to_shape(m, e.shape[2:])
        # decoder
        d = self.dec_layer(torch.cat((m, e), dim=1))
        # output
        s = self.side_layer(d)
        # done!
        return d, [s] + S


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
        return d, [s]


# ========================================================================= #
# Models                                                                    #
# ========================================================================= #


@dataclass
class U2NetLayerCfg:
    # side layer
    side_in_chn: Optional[int]
    # rsu layer
    rsu_height: int
    rsu_in_ch: int
    rsu_mid_ch: int
    rsu_out_ch: int
    rsu_dilated: bool = False

    @classmethod
    def normalise(cls, *layers) -> List['U2NetLayerCfg']:
        return [(layer if isinstance(layer, cls) else cls(**layer)) for layer in layers]


class U2Net(nn.Module):
    """
    Effectively a UNET of differing UNETs (RSU) acting at different scales.
    """

    def __init__(
        self,
        enc_cfgs: List[Union[U2NetLayerCfg, dict]],
        dec_cfgs: List[Union[U2NetLayerCfg, dict]],
        out_ch: int,
        out_activation: str = 'sigmoid',
    ):
        super().__init__()
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # normalise configs -- layers should be pass in feed forward order, eg. encoders: 0, 1, 2, 3, 4, 5 and eg. decoders: 4, 3, 2, 1, 0
        (cfg_mid, *cfgs_encs) = U2NetLayerCfg.normalise(*enc_cfgs)[::-1]  # [..., 5, 4, 3, 2, 1, 0]
        (cfgs_decs)           = U2NetLayerCfg.normalise(*dec_cfgs)        # [...,    4, 3, 2, 1, 0]
        assert len(cfgs_encs) == len(cfgs_decs)
        # vars
        self.out_ch = out_ch
        self.out_activation = out_activation
        self.num_stages = len(cfgs_decs) + 1
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # make layer fns
        def rsu(cfg): return Rsu(num_layers=cfg.rsu_height, in_ch=cfg.rsu_in_ch, mid_ch=cfg.rsu_mid_ch, out_ch=cfg.rsu_out_ch, dilated=cfg.rsu_dilated)
        def out(cfg): return nn.Conv2d(in_channels=cfg.side_in_chn, out_channels=self.out_ch, kernel_size=3, padding=1)
        # create stages
        self.unet = MidStage(mid_layer=rsu(cfg_mid), side_layer=out(cfg_mid))
        for enc_cfg, dec_cfg in zip(cfgs_encs, cfgs_decs):
            self.unet = EncDecStage(enc_layer=rsu(enc_cfg), dec_layer=rsu(dec_cfg), side_layer=out(dec_cfg), child=self.unet)
        # output layer
        self.outconv = nn.Conv2d(in_channels=self.num_stages * self.out_ch, out_channels=self.out_ch, kernel_size=1)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    def forward(self, x):
        # feed forward through unet and get side outputs
        _, outputs = self.unet(x)
        # generate outputs:
        # | 1. rescale
        outputs = [_upsample_to_shape(s, size=x.shape[2:]) for s in outputs]
        # | 2. concatenate
        outputs = [self.outconv(torch.cat(outputs, dim=1)), *outputs]
        # | 3. activate
        if self.out_activation is not None:
            outputs = [activate(s, self.out_activation) for s in outputs]
        # done!
        return outputs


def U2NET_full(out_ch: int = 1, cls=U2Net):
    enc_layer_cfgs = [
        U2NetLayerCfg(rsu_height=7, rsu_in_ch=3,    rsu_mid_ch=32,  rsu_out_ch=64,  rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=6, rsu_in_ch=64,   rsu_mid_ch=32,  rsu_out_ch=128, rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=5, rsu_in_ch=128,  rsu_mid_ch=64,  rsu_out_ch=256, rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=256,  rsu_mid_ch=128, rsu_out_ch=512, rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=512,  rsu_mid_ch=256, rsu_out_ch=512, rsu_dilated=True,  side_in_chn=None),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=512,  rsu_mid_ch=256, rsu_out_ch=512, rsu_dilated=True,  side_in_chn=512),
    ]
    dec_layer_cfgs = [
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=1024, rsu_mid_ch=256, rsu_out_ch=512, rsu_dilated=True,  side_in_chn=512),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=1024, rsu_mid_ch=128, rsu_out_ch=256, rsu_dilated=False, side_in_chn=256),
        U2NetLayerCfg(rsu_height=5, rsu_in_ch=512,  rsu_mid_ch=64,  rsu_out_ch=128, rsu_dilated=False, side_in_chn=128),
        U2NetLayerCfg(rsu_height=6, rsu_in_ch=256,  rsu_mid_ch=32,  rsu_out_ch=64,  rsu_dilated=False, side_in_chn=64),
        U2NetLayerCfg(rsu_height=7, rsu_in_ch=128,  rsu_mid_ch=16,  rsu_out_ch=64,  rsu_dilated=False, side_in_chn=64),
    ]
    return cls(enc_cfgs=enc_layer_cfgs, dec_cfgs=dec_layer_cfgs, out_ch=out_ch)


def U2NET_lite(out_ch: int = 1, cls=U2Net):
    enc_layer_cfgs = [
        U2NetLayerCfg(rsu_height=7, rsu_in_ch=3,  rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=6, rsu_in_ch=64, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=5, rsu_in_ch=64, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=64, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=None),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=64, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=True,  side_in_chn=None),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=64, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=True,  side_in_chn=64),
    ]
    dec_layer_cfgs = [
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=128, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=True,  side_in_chn=64),
        U2NetLayerCfg(rsu_height=4, rsu_in_ch=128, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=64),
        U2NetLayerCfg(rsu_height=5, rsu_in_ch=128, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=64),
        U2NetLayerCfg(rsu_height=6, rsu_in_ch=128, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=64),
        U2NetLayerCfg(rsu_height=7, rsu_in_ch=128, rsu_mid_ch=16, rsu_out_ch=64, rsu_dilated=False, side_in_chn=64),
    ]
    return cls(enc_cfgs=enc_layer_cfgs, dec_cfgs=dec_layer_cfgs, out_ch=out_ch)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':

    x = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    model = U2NET_full()
    outs = model.forward(x)

    print([float(o.mean()) for o in outs])
    print([float(o.std()) for o in outs])
    print([tuple(o.shape) for o in outs])
