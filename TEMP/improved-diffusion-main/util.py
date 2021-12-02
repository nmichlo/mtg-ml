import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence


@dataclass
class Cfg(Mapping):

    # TODO: move this into disent!
    #      this is actually pretty cool

    # TODO: make sure everything is json serializable!

    def to_dict(self) -> dict:
        # this does a lot of extra work!
        # - recursively visits sub-classes
        # - calls deep copy on everything
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def add_parser_args(cls, parser=None, default_args: Optional[Sequence] = None, default_kwargs: Optional[Dict[str, Any]] = None):
        from improved_diffusion.script_util import add_dict_to_argparser
        # get default parser
        if parser is None:
            parser = argparse.ArgumentParser()
        # get default args & kwargs
        if default_args is None: default_args = []
        if default_kwargs is None: default_kwargs = {}
        # instantiate & add defaults to parser
        cfg = cls(*default_args, **default_kwargs)
        add_dict_to_argparser(parser, cfg.to_dict())
        return parser

    @classmethod
    def parse_args(cls, parser=None, default_args: Optional[Sequence] = None, default_kwargs: Optional[Dict[str, Any]] = None) -> 'Cfg':
        parser = cls.add_parser_args(parser=parser, default_args=default_args, default_kwargs=default_kwargs)
        args = parser.parse_args()  # Namespace
        return cls(**args.__dict__)

    @classmethod
    def from_dict(cls, dct):
        if isinstance(dct, argparse.Namespace):
            dct = argparse.__dict__
        return cls(**{
            k: dct[k]
            for k in dataclasses.fields(cls)
        })

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
