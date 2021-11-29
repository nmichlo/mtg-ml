import os
import re
import warnings
from pathlib import Path
from typing import Optional


def _get_data_root(key: str, default_path: str) -> Path:
    # read the environment variable
    path = os.environ.get(key, default_path)
    # check the the path is absolute
    if not os.path.isabs(path):
        warnings.warn(f'path is relative, this may be incorrect! Converting {key}={repr(path)} to absolute path: {repr(os.path.abspath(path))}')
        path = os.path.abspath(path)
    # check that the directory exists
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f'path exists but is not a directory! {key}={repr(path)}')
    else:
        warnings.warn(f'path to directory does not exist, creating it... {key}={repr(path)}')
        os.makedirs(path, exist_ok=True)
    # return the absolute path
    return Path(path)


def get_next_experiment_number(out_root_dir: Optional[str] = None) -> str:
    # get default output folder
    if out_root_dir is None:
        from mtg_ml import OUT_ROOT
        out_root_dir = OUT_ROOT
    # check that the directory exists
    if not os.path.exists(out_root_dir):
        raise FileNotFoundError(f'The directory does not exist: {repr(out_root_dir)}')
    elif not os.path.isdir(out_root_dir):
        raise NotADirectoryError(f'The given path is not a directory: {repr(out_root_dir)}')
    # get next experiment number
    max_number = 0
    root = Path(out_root_dir)
    for subfile in root.glob('*'):
        if not subfile.is_dir():
            continue
        match = re.match('([\d]+)_.*', subfile.name)
        if not match:
            continue
        (num,) = match.groups()
        max_number = max(max_number, int(num))
    # done!
    return f'{max_number + 1:05d}'
