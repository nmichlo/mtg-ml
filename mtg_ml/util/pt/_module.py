from contextlib import contextmanager

import torch


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


@contextmanager
def evaluate_context(module: torch.nn.Module, train: bool = False):
    """
    Temporarily switch a model to evaluation
    mode, and restore the mode afterwards!
    """
    was_training = module.training
    try:
        module.train(mode=train)
        yield module
    finally:
        module.train(mode=was_training)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
