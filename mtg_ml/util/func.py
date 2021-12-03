from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from omegaconf import DictConfig


_INSTANCE_OF_TYPE = Optional[Union[Tuple[Type, ...], Type]]


def _check_instance_of(result: Any, instance_of: _INSTANCE_OF_TYPE) -> Any:
    # check results
    if instance_of is not None:
        if not isinstance(result, instance_of):
            raise ValueError(f'invalid instantiated object, incorrect type: {type(result)}, must be: {instance_of}')
    # return result
    return result


def instantiate_if_needed(value: Any, instance_of: _INSTANCE_OF_TYPE = None) -> Any:
    if isinstance(value, (dict, DictConfig)):
        if '_target_' in value:
            import hydra
            return hydra.utils.instantiate(value)
    # check the result
    return _check_instance_of(value, instance_of=instance_of)


def instantiate_or_none(value: Optional[Dict[str, Any]], instance_of: _INSTANCE_OF_TYPE = None) -> Optional[Any]:
    if value is None:
        return None
    # instantiate and check
    return instantiate_required(value, instance_of=instance_of)


def instantiate_required(value: Dict[str, Any], instance_of: _INSTANCE_OF_TYPE = None) -> Any:
    if isinstance(value, (dict, DictConfig)):
        import hydra
        result = hydra.utils.instantiate(value)
    else:
        raise TypeError(f'invalid config target. Must be a dictionary, got: {repr(value)}')
    # check the result
    return _check_instance_of(result, instance_of=instance_of)
