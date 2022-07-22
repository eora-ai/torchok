"""TorchOK Model creation / weight loading / state_dict helpers.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/helpers.py

Copyright 2020 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Any, Callable, Dict

import torch.nn as nn
from torch.hub import load_state_dict_from_url


def build_model_with_cfg(model_cls: Callable,
                         pretrained: bool,
                         default_cfg: Dict[str, Any],
                         model_cfg: Dict[str, Any] = None,
                         **model_args) -> nn.Module:
    """Build model with specified default_cfg and optional model_args.

    Args:
        model_cls: model class
        pretrained: load pretrained weights
        default_cfg: model's default pretrained/task config
        model_cfg: Configuration for creating the model.
        **model_args: model args passed through to model __init__
    """
    model = model_cls(**model_args) if model_cfg is None else model_cls(cfg=model_cfg, **model_args)
    pretrained_url = default_cfg.get('url', None)

    if pretrained and pretrained_url is not None:
        state_dict = load_state_dict_from_url(pretrained_url, progress=False, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

    return model
