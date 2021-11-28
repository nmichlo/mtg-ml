

def _get_data_root(default_path='data'):
    import os
    import warnings
    from pathlib import Path
    # read the environment variable
    path = os.environ.get('ML_DATA_ROOT', default_path)
    # check the the path is absolute
    if not os.path.isabs(path):
        warnings.warn(f'data root is a relative path, this may be incorrect! Converting ML_DATA_ROOT={repr(path)} to absolute path: {repr(os.path.abspath(path))}')
        path = os.path.abspath(path)
    # check that the directory exists
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f'data root exists but is not a directory! ML_DATA_ROOT={repr(path)}')
    else:
        warnings.warn(f'data root does not exist, creating it... ML_DATA_ROOT={repr(path)}')
        os.makedirs(path, exist_ok=True)
    # return the absolute path
    return Path(path)


# get the root data directory
DATA_ROOT = _get_data_root()

# get the root dataset directory
DATASET_ROOT = DATA_ROOT.joinpath('dataset')
