
from mtg_ml.util.path import _get_data_root

# get the root data directory
DATA_ROOT = _get_data_root('ML_DATA_ROOT', 'data')

# get the root dataset directory
DATASET_ROOT = DATA_ROOT.joinpath('dataset')

# get the output root
ML_OUT_ROOT = _get_data_root('ML_OUT_ROOT', 'out')
