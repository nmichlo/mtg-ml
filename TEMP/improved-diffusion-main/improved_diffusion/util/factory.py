import math
from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from improved_diffusion.util import gaussian_diffusion as gd
from improved_diffusion.util.resample import LossSecondMomentResampler
from improved_diffusion.util.resample import ScheduleSampler
from improved_diffusion.util.resample import UniformSampler
from improved_diffusion.util.respace import space_timesteps
from improved_diffusion.util.respace import SpacedDiffusion
from improved_diffusion.util.unet import SuperResModel
from improved_diffusion.util.unet import UNetModel


# ========================================================================= #
# Model Factory                                                             #
# ========================================================================= #


@dataclass
class _ModelCfg:
    # both
    learn_sigma: bool = False
    # model
    image_size: int = 64
    num_channels: int = 128
    num_res_blocks: int = 2
    num_heads: int = 4
    num_heads_upsample: int = -1
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)  # {256: (1, 1, 2, 2, 4, 4), 64: (1, 2, 3, 4), 32: (1, 2, 2, 2)}[image_size]
    attention_resolutions: str = "16,8"
    dropout: float = 0.0
    num_classes: Optional[int] = None  # NUM_CLASSES=1000
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True
    # not part of config
    _model_cls_ = None


@dataclass
class ImageModelCfg(_ModelCfg):
    # not part of config
    _model_cls_ = UNetModel


@dataclass
class SrModelCfg(_ModelCfg):
    image_size: int = 256  # was: large_size
    small_size: int = 64
    channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4, 4)  # {256: (1, 1, 2, 2, 4, 4), 64: (1, 2, 3, 4), 32: (1, 2, 2, 2)}[image_size]
    # not part of config
    _model_cls_ = SuperResModel


def create_model(cfg: Union[ImageModelCfg, SrModelCfg]) -> Union[UNetModel, SuperResModel]:
    return cfg._model_cls_(
        in_channels=3,
        model_channels=cfg.num_channels,
        out_channels=(3 if not cfg.learn_sigma else 6),
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=[cfg.image_size // int(res) for res in cfg.attention_resolutions.split(",")],
        dropout=cfg.dropout,
        channel_mult=cfg.channel_mult,
        num_classes=cfg.num_classes,
        use_checkpoint=cfg.use_checkpoint,
        num_heads=cfg.num_heads,
        num_heads_upsample=cfg.num_heads_upsample,
        use_scale_shift_norm=cfg.use_scale_shift_norm,
    )


# ========================================================================= #
# Diffusion Factory                                                         #
# ========================================================================= #


@dataclass
class DiffusionAndSampleCfg:
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


def create_diffusion_and_sampler(cfg: DiffusionAndSampleCfg) -> Tuple[gd.GaussianDiffusion, ScheduleSampler]:
    # TODO: sigma_small should be False if we are using the Sr Model

    betas = get_named_beta_schedule(cfg.noise_schedule, cfg.diffusion_steps)

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

# ========================================================================= #
# Sampler Factory                                                           #
# ========================================================================= #


def create_named_schedule_sampler(name: str, num_timesteps: int):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(num_timesteps=num_timesteps)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(num_timesteps=num_timesteps)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")

# ========================================================================= #
# Schedule Factory                                                          #
# ========================================================================= #


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        from improved_diffusion.util.gaussian_diffusion import betas_for_alpha_bar
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
