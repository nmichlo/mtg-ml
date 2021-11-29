
import logging
import os
from datetime import datetime

import pytorch_lightning as pl
import hydra
import wandb
from disent.util.seeds import seed
from disent.util.strings.fmt import make_box_str
from omegaconf import DictConfig
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


def action_train(cfg: DictConfig):

    # get the time the run started
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    logger.info(f'Starting run at time: {time_string}')

    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # cleanup from old runs:
    try:
        wandb.finish()
    except:
        pass

    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # deterministic seed
    seed(cfg.settings.job.seed)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # INITIALISE & SETDEFAULT IN CONFIG
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # print useful info
    logger.info(f"Current working directory : {os.getcwd()}")
    logger.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # print config sections
    logger.info(f'Final Config:\n\n{make_box_str(OmegaConf.to_yaml(cfg))}')

    # HYDRA MODULES
    datamodule = hydra.utils.instantiate(cfg.data.module_cls, _recursive_=False)
    framework = hydra.utils.instantiate(cfg.framework.system_cls)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # BEGIN TRAINING
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # save hparams TODO: is this a pytorch lightning bug? The trainer should automatically save these if hparams is set?
    # framework.hparams.update(cfg)
    # if trainer.logger:
    #     trainer.logger.log_hyperparams(framework.hparams)

    # Setup Trainer
    trainer = pl.Trainer(**cfg.trainer)

    # fit the model
    trainer.fit(framework, datamodule)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # cleanup this run
    try:
        wandb.finish()
    except:
        pass

    # -~-~-~-~-~-~-~-~-~-~-~-~- #


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


# path to root directory containing configs
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config'))
# root config existing inside `CONFIG_ROOT`, with '.yaml' appended.
CONFIG_NAME = 'config'


if __name__ == '__main__':

    # register a custom OmegaConf resolver that allows us to put in a ${exit:msg} that exits the program
    # - if we don't register this, the program will still fail because we have an unknown
    #   resolver. This just prettifies the output.
    class ConfigurationError(Exception):
        pass
    def _error_resolver(msg: str):
        raise ConfigurationError(msg)
    OmegaConf.register_new_resolver('exit', _error_resolver)

    # main function
    @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
    def hydra_main(cfg: DictConfig):
        try:
            action_train(cfg)
        except Exception as e:
            logger.error(f'cfg={cfg}')
            logger.error(f'experiment - error: {e}', exc_info=True)

    # entrypoint
    try:
        hydra_main()
    except KeyboardInterrupt as e:
        logger.warning(f'hydra - interrupted: {e}', exc_info=False)
    except Exception as e:
        logger.error(f'hydra - error: {e}', exc_info=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
