from typing import Any
from typing import Dict
from typing import Optional

from omegaconf import DictConfig


def instantiate_if_needed(value: Any) -> Any:
    if isinstance(value, (dict, DictConfig)):
        if '_target_' in value:
            import hydra
            return hydra.utils.instantiate(value)
    return value


def instantiate_or_none(value: Optional[Dict[str, Any]]) -> Optional[Any]:
    if value is None:
        return None
    elif isinstance(value, (dict, DictConfig)):
        import hydra
        return hydra.utils.instantiate(value)
    else:
        raise TypeError(f'invalid config target. Must be None or a dictionary, got: {repr(value)}')
