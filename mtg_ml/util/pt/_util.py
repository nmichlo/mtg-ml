from typing import Optional

import numpy as np
import torch


# ========================================================================= #
# Params                                                                    #
# ========================================================================= #


def count_params(model: Optional[torch.nn.Module], mode: str = 'all') -> int:
    # handle a missing model
    if model is None:
        return 0
    # handle the modes
    if mode == 'all':
        return sum(p.numel() for p in model.parameters())
    elif mode == 'trainable':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == 'frozen':
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)
    else:
        raise KeyError('invalid param count mode')


def count_params_pretty(model: Optional[torch.nn.Module], mode: str = 'all') -> str:
    p = count_params(model, mode=mode)
    pow = 0 if (p == 0) else int(np.log(p) / np.log(1000))
    mul = 1000 ** pow
    symbol = {0: '', 1: 'K', 2: 'M', 3: 'B', 4: 'T', 5: 'P', 6: 'E'}[pow]
    return f'{p/mul:5.1f}{symbol}'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
