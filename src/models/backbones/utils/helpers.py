"""TorchOK Model creation / weight loading / state_dict helpers.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/helpers.py

Copyright 2020 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
import logging
import math
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

_logger = logging.getLogger(__name__)


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


def load_pretrained(model: nn.Module,
                    in_chans: int = 3,
                    strict: bool = True,
                    progress: bool = False) -> None:
    """Load pretrained checkpoint.

    Args:
        model: PyTorch model module.
        in_chans: Input channels for model.
        strict: Strict load of checkpoint.
        progress: Enable progress bar for weight download.
    """
    default_cfg = getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)

    if not pretrained_url:
        _logger.warning('No pretrained weights exist for this model. Using random initialization.')
        return

    _logger.info(f'Loading pretrained weights from url ({pretrained_url})')

    state_dict = load_state_dict_from_url(pretrained_url, progress=progress, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    input_convs = default_cfg.get('first_conv', None)

    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifier_name = default_cfg.get('classifier', None)
    if classifier_name is not None:
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']

    model.load_state_dict(state_dict, strict=strict)


def build_model_with_cfg(model_cls: Callable,
                         pretrained: bool,
                         default_cfg: dict,
                         **model_args) -> nn.Module:
    """Build model with specified default_cfg and optional model_args.

    Args:
        model_cls: model class
        pretrained: load pretrained weights
        default_cfg: model's default pretrained/task config
        **model_args: model args passed through to model __init__
    """
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    model = model_cls(**model_args)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model,
                        in_chans=model_args.get('in_chans', 3),
                        strict=True)
    return model
