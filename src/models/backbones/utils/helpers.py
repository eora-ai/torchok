"""TorchOK Model creation / weight loading / state_dict helpers.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/helpers.py

Copyright 2020 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
import math
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


def adapt_input_conv(input_data_channels: int,
                     conv_weight: torch.Tensor):
    """Adaptation input conv block to input data.

    Args:
        input_data_channels: Number of input data channels.
        conv_weight: Weights of conv block.
    """
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    weight_out_channels, weight_input_channels, weight_kernel_h, weight_kernel_w = conv_weight.shape

    if input_data_channels == 1:
        if weight_input_channels > 3:
            assert weight_input_channels % 3 == 0
            conv_weight = conv_weight.reshape(weight_out_channels,
                                              weight_input_channels // 3,
                                              3,
                                              weight_kernel_h,
                                              weight_kernel_w)

            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif input_data_channels != 3:
        if weight_input_channels != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            repeat = math.ceil(input_data_channels / 3)
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :input_data_channels, :, :]
            conv_weight *= (3 / float(input_data_channels))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def build_model_with_cfg(model_cls: Callable,
                         pretrained: bool,
                         default_cfg: Dict[str, Any],
                         **model_args) -> nn.Module:
    """Build model with specified default_cfg and optional model_args.

    Args:
        model_cls: model class
        pretrained: load pretrained weights
        default_cfg: model's default pretrained/task config
        **model_args: model args passed through to model __init__
    """
    model = model_cls(**model_args)
    pretrained_url = default_cfg.get('url', None)
    in_chans = model_args.get('in_chans', 3)

    if pretrained and pretrained_url is not None:
        state_dict = load_state_dict_from_url(pretrained_url, progress=False, map_location='cpu')
        input_convs = default_cfg.get('first_conv', None)
        if input_convs is not None and in_chans != 3:
            state_dict[input_convs] = adapt_input_conv(in_chans, state_dict[input_convs])

    model.load_state_dict(state_dict, strict=True)
    return model
