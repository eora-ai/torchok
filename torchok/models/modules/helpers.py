"""TorchOK Layer/Module Helpers

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
Copyright 2020 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
