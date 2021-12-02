import argparse
from dataclasses import dataclass
from typing import Union

from util import Cfg
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel

NUM_CLASSES = 1000


@dataclass
class _ModelDiffusionCfg(Cfg):
    image_size: int = 64
    num_channels: int = 128
    num_res_blocks: int = 2
    num_heads: int = 4
    num_heads_upsample: int = -1
    attention_resolutions: str = "16,8"
    dropout: float = 0.0
    learn_sigma: bool = False
    sigma_small: bool = False
    class_cond: bool = False
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = ""
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = True
    rescale_learned_sigmas: bool = True
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True


@dataclass
class ImageModelDiffusionCfg(_ModelDiffusionCfg):
    pass


@dataclass
class SrModelDiffusionCfg(_ModelDiffusionCfg):
    image_size: int = 256  # was: large_size
    small_size: int = 64


def create_model_and_diffusion(cfg: Union[ImageModelDiffusionCfg, SrModelDiffusionCfg]):
    # checks
    if isinstance(cfg, SrModelDiffusionCfg):
        assert cfg.sigma_small is False
        model_cls = SuperResModel
    elif isinstance(cfg, ImageModelDiffusionCfg):
        model_cls = UNetModel
    else:
        raise TypeError(f'cfg must be of type {ImageModelDiffusionCfg.__name__} or {SrModelDiffusionCfg.__name__}, got: {type(cfg)}')

    # make model
    model = create_model(
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
        learn_sigma=cfg.learn_sigma,
        class_cond=cfg.class_cond,
        use_checkpoint=cfg.use_checkpoint,
        attention_resolutions=cfg.attention_resolutions,
        num_heads=cfg.num_heads,
        num_heads_upsample=cfg.num_heads_upsample,
        use_scale_shift_norm=cfg.use_scale_shift_norm,
        dropout=cfg.dropout,
        model_cls=model_cls,
    )

    # make diffusion
    diffusion = create_gaussian_diffusion(
        steps=cfg.diffusion_steps,
        learn_sigma=cfg.learn_sigma,
        sigma_small=cfg.sigma_small,
        noise_schedule=cfg.noise_schedule,
        use_kl=cfg.use_kl,
        predict_xstart=cfg.predict_xstart,
        rescale_timesteps=cfg.rescale_timesteps,
        rescale_learned_sigmas=cfg.rescale_learned_sigmas,
        timestep_respacing=cfg.timestep_respacing,
    )

    return model, diffusion


_DEFAULT_CHANNELS = {
    256: (1, 1, 2, 2, 4, 4),
    64: (1, 2, 3, 4),
    32: (1, 2, 2, 2),
}


def _get_channels(image_size: int):
    if image_size not in _DEFAULT_CHANNELS:
        raise ValueError(f"unsupported image size: {image_size}")
    return _DEFAULT_CHANNELS[image_size]


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    model_cls,
):
    channel_mult = _get_channels(image_size=image_size)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return model_cls(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


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
