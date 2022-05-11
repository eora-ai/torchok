""" TorchOK Model creation / weight loading / state_dict helpers

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/helpers.py

Copyright 2020 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info(f"Loaded {state_dict_key} from checkpoint '{checkpoint_path}'")
        return state_dict
    else:
        _logger.error(f"No checkpoint found at '{checkpoint_path}'")
        raise FileNotFoundError()


def adapt_input_conv(in_chans: int, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained(model: nn.Module,
                    default_cfg: Optional[Dict] = None,
                    in_chans: int = 3,
                    filter_fn: Optional[Callable] = None,
                    strict: bool = False,
                    progress: bool = False) -> None:
    """ Load pretrained checkpoint

    Args:
        model: PyTorch model module
        default_cfg: default configuration for pretrained weights / target dataset
        in_chans: in_chans for model
        filter_fn: state_dict filter fn for load (takes state_dict, model as args)
        strict: strict load of checkpoint
        progress: enable progress bar for weight download

    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    if not pretrained_url:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return

    _logger.info(f'Loading pretrained weights from url ({pretrained_url})')
    state_dict = load_state_dict_from_url(pretrained_url, progress=progress, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

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
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifier_name = default_cfg.get('classifier', None)
    if classifier_name is not None:
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']

    model.load_state_dict(state_dict, strict=strict)

def overlay_external_default_cfg(default_cfg: Optional[Dict],
                                 kwargs: Dict):
    """ Overlay 'external_default_cfg' in kwargs on top of default_cfg arg."""
    external_default_cfg = kwargs.pop('external_default_cfg', None)
    if external_default_cfg:
        default_cfg.pop('url', None)  # url should come from external cfg
        default_cfg.pop('hf_hub', None)  # hf hub id should come from external cfg
        default_cfg.update(external_default_cfg)


def set_default_kwargs(kwargs: dict,
                       names: Optional[Tuple[str]],
                       default_cfg: dict):
    for n in names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # default_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = default_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, default_cfg[n])


def filter_kwargs(kwargs: dict,
                  names: Optional[Tuple[str]]) -> None:
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def update_default_cfg_and_kwargs(default_cfg: dict,
                                  kwargs: dict,
                                  kwargs_filter: Optional[Tuple[str]]) -> None:
    """ Update the default_cfg and kwargs before passing to model

    Args:
        default_cfg: input default_cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Overlay default cfg values from `external_default_cfg` if it exists in kwargs
    overlay_external_default_cfg(default_cfg, kwargs)
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    set_default_kwargs(kwargs, names=('num_classes', 'global_pool', 'in_chans'), default_cfg=default_cfg)
    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    filter_kwargs(kwargs, names=kwargs_filter)


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: Optional[Any] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs) -> nn.Module:
    """Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation

    Args:
        model_cls: model class
        variant: model variant name
        pretrained: load pretrained weights
        default_cfg: model's default pretrained/task config
        model_cfg: model's architecture config
        pretrained_strict: load pretrained weights strictly
        pretrained_filter_fn: filter callable for pretrained weights
        kwargs_filter: kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter)
    default_cfg.setdefault('architecture', variant)

    # Build the model
    kwargs.pop('num_classes', False)
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model,
                        in_chans=kwargs.get('in_chans', 3),
                        filter_fn=pretrained_filter_fn,
                        strict=pretrained_strict)
    return model
