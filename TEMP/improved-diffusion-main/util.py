import argparse
from dataclasses import dataclass
from typing import Mapping


@dataclass
class Cfg(Mapping):

    def to_dict(self) -> dict:
        # this does a lot of extra work!
        # - recursively visits sub-classes
        # - calls deep copy on everything
        from dataclasses import asdict
        return asdict(self)

    def to_arg_parser(self, parser=None):
        from improved_diffusion.script_util import add_dict_to_argparser
        if parser is None:
            parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, self.to_dict())
        return parser

    @classmethod
    def parse_args(cls, *default_args, parser=None, **default_kwargs) -> 'Cfg':
        cfg = cls(*default_args, **default_kwargs)
        parser = cfg.to_arg_parser(parser=parser)
        args = parser.parse_args()  # Namespace
        return cls(**args.__dict__)

    def __str__(self):
        from pprint import pformat
        return pformat(self.to_dict(), sort_dicts=False)

    # Mapping

    def __getitem__(self, key: str):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
