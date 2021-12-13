import logging
import os
import pickle
from datetime import datetime
from functools import wraps

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

# avoid importing torch here


logger = logging.getLogger(__name__)


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def wandb_cleanup(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # cleanup from old runs:
        try:
            import wandb
            wandb.finish()
        except:
            pass
        # wrap function
        result = func(*args, **kwargs)
        # cleanup this run
        try:
            import wandb
            wandb.finish()
        except:
            pass
        # done!
        return result
    return wrapper


# ========================================================================= #
# ACTION                                                                    #
# ========================================================================= #


@wandb_cleanup
def action_train(cfg: DictConfig):
    # import torch here
    from disent.util.seeds import seed
    from disent.util.strings.fmt import make_box_str

    # get the time the run started
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    logger.info(f'Starting run at time: {time_string}')
    # print useful info
    logger.info(f"Current working directory : {os.getcwd()}")
    logger.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    # print config sections
    logger.info(f'Config:\n{make_box_str(OmegaConf.to_yaml(cfg))}')

    # SEED
    seed(cfg.settings.job.seed)

    # HYDRA MODULES
    datamodule = hydra.utils.instantiate(cfg.data.module_cls)
    framework = hydra.utils.instantiate(cfg.framework.system_cls)

    pickle.dumps(datamodule)
    pickle.dumps(framework)

    # TRAIN
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(framework, datamodule)


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
    def _exp_num_resolver(path: str):
        from mtg_ml.util.path import get_next_experiment_number
        return get_next_experiment_number(path)
    OmegaConf.register_new_resolver('exit', _error_resolver)
    OmegaConf.register_new_resolver('next_num', _exp_num_resolver)

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
