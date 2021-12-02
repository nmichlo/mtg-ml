from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import Union

from . import gaussian_diffusion as gd
from .gaussian_diffusion import GaussianDiffusion
from .resample import create_named_schedule_sampler
from .resample import ScheduleSampler
from .respace import space_timesteps
from .respace import SpacedDiffusion
from .unet import SuperResModel
from .unet import UNetModel


NUM_CLASSES = 1000


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
