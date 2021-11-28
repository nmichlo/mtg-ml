from types import FunctionType, BuiltinFunctionType
import torch


_NORM_CONSTS = {}


def norm_activation(x, activation: callable, samples=16384):
    # get the activation key
    if isinstance(activation, torch.nn.Module):
        key = activation.__class__
    elif isinstance(activation, (FunctionType, BuiltinFunctionType)):
        key = activation
    else:
        raise TypeError(f'invalid activation type: {activation}')
    # compute the activation if it does not exist
    if key not in _NORM_CONSTS:
        act = activation(torch.randn(samples))
        _NORM_CONSTS[key] = (act.mean().item(), act.std().item())
    # get the normalisation constants
    mean, std = _NORM_CONSTS[key]
    # activate and normalise!
    return (activation(x) - mean) / std
