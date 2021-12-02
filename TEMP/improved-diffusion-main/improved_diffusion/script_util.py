import argparse
from dataclasses import dataclass
from typing import Tuple
from typing import Union

from util import Cfg
from . import gaussian_diffusion as gd
from .gaussian_diffusion import GaussianDiffusion
from .resample import create_named_schedule_sampler
from .resample import ScheduleSampler
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel


NUM_CLASSES = 1000


@dataclass
class _ModelCfg(Cfg):
    # both
    learn_sigma: bool = False
    # model
    image_size: int = 64
    num_channels: int = 128
    num_res_blocks: int = 2
    num_heads: int = 4
    num_heads_upsample: int = -1
    attention_resolutions: str = "16,8"
    dropout: float = 0.0
    class_cond: bool = False
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True


@dataclass
class ImageModelCfg(_ModelCfg):
    pass


@dataclass
class SrModelCfg(_ModelCfg):
    image_size: int = 256  # was: large_size
    small_size: int = 64


@dataclass
class DiffusionAndSampleCfg(Cfg):
    # both
    learn_sigma: bool = False
    # diffusion
    diffusion_steps: int = 1000
    sigma_small: bool = False
    noise_schedule: str = "linear"
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = True       # might actually be false?
    rescale_learned_sigmas: bool = True  # might actually be false?
    timestep_respacing: str = ""
    # schedule sampler
    schedule_sampler: str = "uniform"


_DEFAULT_CHANNELS = {
    256: (1, 1, 2, 2, 4, 4),
    64: (1, 2, 3, 4),
    32: (1, 2, 2, 2),
}


def _get_channels(image_size: int):
    if image_size not in _DEFAULT_CHANNELS:
        raise ValueError(f"unsupported image size: {image_size}")
    return _DEFAULT_CHANNELS[image_size]


def create_model(cfg: Union[ImageModelCfg, SrModelCfg]):
    # checks
    if isinstance(cfg, SrModelCfg):
        model_cls = SuperResModel
    elif isinstance(cfg, ImageModelCfg):
        model_cls = UNetModel
    else:
        raise TypeError(f'cfg must be of type {ImageModelCfg.__name__} or {SrModelCfg.__name__}, got: {type(cfg)}')

    channel_mult = _get_channels(image_size=cfg.image_size)

    attention_ds = []
    for res in cfg.attention_resolutions.split(","):
        attention_ds.append(cfg.image_size // int(res))

    return model_cls(
        in_channels=3,
        model_channels=cfg.num_channels,
        out_channels=(3 if not cfg.learn_sigma else 6),
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=cfg.dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if cfg.class_cond else None),
        use_checkpoint=cfg.use_checkpoint,
        num_heads=cfg.num_heads,
        num_heads_upsample=cfg.num_heads_upsample,
        use_scale_shift_norm=cfg.use_scale_shift_norm,
    )


def create_diffusion_and_sampler(cfg: DiffusionAndSampleCfg) -> Tuple[GaussianDiffusion, ScheduleSampler]:
    # TODO: sigma_small should be False if we are using the Sr Model

    betas = gd.get_named_beta_schedule(cfg.noise_schedule, cfg.diffusion_steps)

    if cfg.use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif cfg.rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    timestep_respacing = [cfg.diffusion_steps] if (not cfg.timestep_respacing) else cfg.timestep_respacing

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(cfg.diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not cfg.predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE if not cfg.sigma_small else gd.ModelVarType.FIXED_SMALL) if not cfg.learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=cfg.rescale_timesteps,
    )

    sampler = create_named_schedule_sampler(
        name=cfg.schedule_sampler,
        num_timesteps=diffusion.num_timesteps,
    )

    return diffusion, sampler


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
