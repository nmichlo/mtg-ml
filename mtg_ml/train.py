import os
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import hydra

from mtg_ml.util.common import Hdf5DataModule


def make_mtg_datamodule(
    h5_path: str,
    # load everything
    batch_size: int = 32,
    num_workers: int = os.cpu_count(),
    val_ratio: float = 0,
    # convert options
    # load_path: str = None,
    # data_root: Optional[str] = None,
    # convert_kwargs: Dict[str, Any] = None,
    # extra data_loader transform
    to_tensor=True,
    transform=None,
    mean_std: Optional[Tuple[float, float]] = None,
    # memory
    in_memory=False,
):
    # from mtgdata.scryfall_convert import generate_converted_dataset

    # generate training set
    # if load_path is None:
    #     if convert_kwargs is None:
    #         convert_kwargs = {}
    #     h5_path, meta_path = generate_converted_dataset(save_root=data_root, data_root=data_root, **convert_kwargs)
    # else:
    #     assert not convert_kwargs, '`convert_kwargs` cannot be set if `data_path` is specified'
    #     assert not data_root, '`data_root` cannot be set if `data_path` is specified'
    #     h5_path = load_path

    # get transform

    return Hdf5DataModule(
        h5_path,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers,
        to_tensor=to_tensor,
        mean_std=mean_std,
        transform=transform,
        in_memory=in_memory,
    )


def make_mtg_trainer(
    # training
    train_epochs: int = None,
    train_steps: int = None,
    cuda: Union[bool, int] = torch.cuda.is_available(),
    # visualise
    visualize_period: int = 500,
    visualize_input: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[float, float]]]]] = None,
    # utils
    checkpoint_period: int = 2500,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_monitor: Optional[str] = 'loss',
    resume_from_checkpoint: str = None,
    # trainer kwargs
    trainer_kwargs: dict = None,
    # logging
    wandb_enabled: bool = False,
    wandb_name: str = None,
    wandb_project: str = None,
    wandb_kwargs: dict = None,
):
    time_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # initialise callbacks
    callbacks = []
    if wandb_enabled:
        callbacks.append(WandbContextManagerCallback())
    if visualize_period and (visualize_input is not None):
        for k, v in visualize_input.items():
            v, mean_std = (v if isinstance(v, tuple) else (v, None))
            callbacks.append(VisualiseCallback(name=k, input_batch=v, every_n_steps=visualize_period, log_wandb=wandb_enabled, log_local=not wandb_enabled, mean_std=mean_std))

    if checkpoint_period:
        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, time_str),
            monitor=checkpoint_monitor,
            every_n_train_steps=checkpoint_period,
            verbose=True,
            save_top_k=None if (checkpoint_monitor is None) else 5,
        ))

    # initialise logger
    logger = True
    if wandb_enabled:
        assert isinstance(wandb_name, str) and wandb_name, f'`wandb_name` must be a non-empty str, got: {repr(wandb_name)}'
        assert isinstance(wandb_project, str) and wandb_project, f'`wandb_project` must be a non-empty str, got: {repr(wandb_project)}'
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(name=f'{time_str}:{wandb_name}', project=wandb_project, **(wandb_kwargs if (wandb_kwargs is not None) else {}))

    # initialise model trainer
    return pl.Trainer(
        gpus=(1 if cuda else 0) if isinstance(cuda, bool) else cuda,
        max_epochs=train_epochs,
        max_steps=train_steps,
        # checkpoint_callback=False,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        callbacks=callbacks,
        weights_summary='full',
        # extra kwargs
        **(trainer_kwargs if trainer_kwargs else {}),
    )

# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #
# from mtg_ml.util.common import make_mtg_trainer

#
# # @hydra.main('')
# # def main()
# from mtg_ml.framework._vae_soft_intro import SoftIntroVaeSystem
#
#
# if __name__ == '__main__':
#
#     def main(data_path: str = None, resume_path: str = None, wandb: bool = False):
#         # settings:
#         # 5878MiB / 5932MiB (RTX2060)
#         system = DCGAN(
#             obs_shape=(3, 224, 160),
#         )
#         vis_input = system.sample_z(8)
#
#         # start training model
#         datamodule = make_mtg_datamodule(
#             batch_size=32,
#             load_path=data_path,
#         )
#
#         trainer = make_mtg_trainer(
#             train_epochs=500,
#             visualize_period=500,
#             resume_from_checkpoint=resume_path,
#             visualize_input={'samples': vis_input},
#             wandb_enabled=wandb,
#             wandb_project='MTG-GAN',
#             wandb_name='MTG-GAN',
#         )
#
#         trainer.fit(system, datamodule)
#
#     # ENTRYPOINT
#     logging.basicConfig(level=logging.INFO)
#     main(
#         data_path='data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
#         resume_path=None,
#         wandb=True,
#     )
#
#
# if __name__ == '__main__':
#
#     def main(data_path: str = None, resume_path: str = None, wandb: bool = True):
#         # get dataset & visualise images
#         datamodule = make_mtg_datamodule(
#             batch_size=32,
#             load_path=data_path,
#         )
#
#         system = SimpleVaeGan(
#             # MEDIUM | batch_size=64 gives ~4070MiB at ~2.47it/s
#             # z_size=256,
#             # hidden_size=384,
#             # dsc_features=make_features(16, 128, num=5),
#             # gen_features=make_features(128, 48, num=5),
#             # gen_features_pix=32,
#
#             # LARGE | batch_size=32 gives ~5530MiB at ~2.08it/s
#             z_size=384,                              # 256
#             hidden_size=512,                         # 512
#             dsc_features=(64, 96, 128, 192, 224),    # (16, 32, 64, 128, 192)
#             gen_features=(256, 224, 192, 128, 96),   # (256, 256, 192, 128, 96)
#             gen_features_pix=64,                     # 64
#
#             # GENERAL
#             share_enc=True,
#             ae_recon_loss='mse_laplace',
#         )
#
#         # start training model
#         trainer = make_mtg_trainer(
#             train_epochs=500,
#             visualize_period=500,
#             visualize_input={
#                 'samples': system.sample_z(8),
#                 'recons': torch.stack([datamodule.data[i] for i in [3466, 18757, 20000, 40000, 21586, 20541, 1100]]),
#             },
#
#             wandb_enabled=wandb,
#             wandb_project='MTG-GAN',
#             wandb_name='MTG-GAN',
#             wandb_kwargs=dict(tags=['large']),
#             checkpoint_monitor='ae_loss_rec',
#             resume_from_checkpoint=resume_path,
#         )
#         trainer.fit(system, datamodule)
#
#     # ENTRYPOINT
#     logging.basicConfig(level=logging.INFO)
#     main(
#         data_path='data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
#         resume_path=None,
#         wandb=True,
#     )
#
#
#
#
# if __name__ == '__main__':
#
#     def main(data_path: str = None, resume_path: str = None):
#
#         # settings:
#         # 5926MiB / 5932MiB (RTX2060)
#
#         print('[initialising]: model')
#         system = MtgVaeSystem(
#             # training options
#             lr=3e-4,
#             alpha=1,
#             beta=0.001,
#             recon_loss='mse',
#
#             # LARGE MODEL ALT: batch_size 32 (2gpus) -- 2*11GB -- 2.0it/s
#             z_size=768,
#             repr_hidden_size=None,  # 1024+128,
#             repr_channels=64,  # 64*7*5 == 2240  # this should probably be increased, but it is slow! maybe continue downsampling beyond 5x7 (64*7*5==2240) -> 2x3 (64*7*5==384)
#             channel_mul=1.26,
#             channel_start=160,
#             channel_dec_mul=1.5,  # enc: 160->320, dec: 240->480
#
#             # MEDIUM MODEL: batch_size 32 -- 5898MiB -- 2.3it/s
#             # z_size=768,
#             # repr_hidden_size=None,  # 1024+128,
#             # repr_channels=64,       # 64*7*5 == 2240
#             # channel_mul=1.245,
#             # channel_start=120,
#             # channel_dec_mul=1.0,  # enc: 120->231, dec: 231->120
#
#             # MEDIUM MODEL ALT: batch_size 32 -- 5898MiB -- 2.3it/s
#             # z_size=768,
#             # repr_hidden_size=None,  # 1024+128,
#             # repr_channels=64,       # 64*7*5 == 2240
#             # channel_mul=1.26,
#             # channel_start=96,
#             # channel_dec_mul=1.3334,  # enc: 96->192, dec: 256->128
#
#             # SMALLER MODEL - batch_size=32 7.02it/s 2738MiB
#             # z_size=256,
#             # repr_hidden_size=None,  # 512,
#             # repr_channels=32,  # 32*5*7 = 1120
#             # channel_mul=1.26,
#             # channel_start=32,
#             # channel_dec_mul=1.5,  # enc: 32->64, dec: 128->64
#
#             # good model defaults
#             model_weight_init=None,
#             model_activation='swish',
#             model_norm='batch',
#             model_skip_mode='inner_some',  # inner_some, inner, all, none
#             model_skip_downsample='ave',
#             model_skip_upsample='bilinear',
#             model_downsample='stride',
#             model_upsample='stride',
#         )
#
#         # get dataset & visualise images
#         print('[initialising]: data')
#         mean_std = (0.5, 0.5)  # TODO: compute actual mean
#         datamodule = make_mtg_datamodule(batch_size=32, load_path=data_path, mean_std=mean_std, in_memory=False, num_workers=32)
#         vis_imgs = torch.stack([datamodule.data[i] for i in [3466, 18757, 20000, 40000, 21586, 20541, 1100]])
#
#         # start training model
#         print('[initialising]: trainer')
#         h = system.hparams
#         trainer = make_mtg_trainer(
#             train_epochs=500,
#             resume_from_checkpoint=resume_path,
#             visualize_period=1000,
#             visualize_input={'recons': (vis_imgs, mean_std)},
#             wandb_project='MTG-VAE',
#             wandb_name=f'mtg-vae|{h.z_size}:{h.repr_hidden_size}:{h.repr_channels}:{h.channel_start}:{h.channel_mul}:{h.channel_dec_mul}',
#             wandb_enabled=True,
#             # distributed
#             trainer_kwargs=dict(accelerator='ddp'),
#             cuda=-1,
#         )
#
#         print('[training]:')
#         trainer.fit(system, datamodule)
#
#     # ENTRYPOINT
#     logging.basicConfig(level=logging.INFO)
#     print('[starting]:')
#     main(
#         data_path='/dev/shm/nmichlo/data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
#         # data_path='/tmp/nmichlo/data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
#         # data_path='data/mtg_default-cards_20210608210352_border-crop_60480x224x160x3_c9.h5',
#         resume_path=None,  # '/home/nmichlo/workspace/playground/mtg-dataset/checkpoints/2021-06-27_23:16:48/epoch=17-step=32499.ckpt',
#     )




if __name__ == '__main__':



    def main(dataset: str = 'mtg', wandb_enabled: bool = False):
        cfg = _SETTINGS[dataset]

        print('[initialising]: model')
        system = SoftIntroVaeSystem(
            dataset=dataset,
            beta_kl=cfg.beta_kl,
            beta_rec=cfg.beta_rec,
            beta_neg=cfg.beta_neg,
            z_size=cfg.z_size,
            lr_dec=2e-4,
            lr_enc=2e-4,
        )

        # get dataset & visualise images
        print('[initialising]: data')
        mean_std = (0.5, 0.5)
        dataset  = DataLoader(system.dataset_settings.make_dataset(), batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count())
        vis_imgs = torch.stack([normalize_image_obs(dataset.dataset[i]) for i in range(5)])

        # start training model
        print('[initialising]: trainer')
        trainer = make_mtg_trainer(
            train_epochs=400,
            visualize_period=500,
            visualize_input={'recons': (vis_imgs, mean_std)},
            checkpoint_monitor=None,
            # wandb settings
            wandb_project='soft-intro-vae',
            wandb_name=f'{dataset}:{cfg.z_size}',
            wandb_enabled=wandb_enabled,
        )

        trainer.fit(system, dataset)

    # ENTRYPOINT
    logging.basicConfig(level=logging.INFO)
    main(
        dataset='mnist',
        wandb_enabled=False,
    )
