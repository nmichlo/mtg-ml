import math
from typing import Optional

from disent.util.math.random import random_choice_prng
from torch.utils.data import Dataset


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


class RandomDataSubset(Dataset):

    def __init__(
        self,
        dataset,
        use_ratio: Optional[float] = None,
        use_number: Optional[int] = None,
        seed: int = 777,
    ):
        # make sure we only use one of the arguments
        if (use_ratio is None) and (use_number is None):
            raise ValueError('one of `use_ratio` or `use_number` must be specified!')
        elif (use_ratio is not None) and (use_number is not None):
            raise ValueError('only one of `use_ratio` or `use_number` must be specified!')
        # checks
        assert len(dataset) > 0, 'the dataset must have at least one element!'
        # compute the dataset size
        if use_ratio is not None:
            assert use_number is None
            assert isinstance(use_ratio, float)
            assert 0 < use_ratio <= 1, f'`use_ratio={repr(use_ratio)}` must be in range (0, 1]'
            use_number = max(1, int(math.ceil(len(dataset) * use_ratio)))
        # check the dataset size
        assert isinstance(use_number, int)
        assert use_number <= len(dataset), f'the `use_number={use_number}` must be <= the dataset size ({len(dataset)})'
        # generate random subset of unique indices
        self._indices = random_choice_prng(len(dataset), size=use_number, replace=False, seed=seed)
        self._dataset = dataset

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._dataset[self._indices[idx]]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
