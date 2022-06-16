"""TorchOK EffcientNet.
Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
import math
from typing import Union, Dict, Any, List

import torch.nn as nn
from torch import Tensor

from src.constructor import BACKBONES
from src.models.base_model import BaseModel
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.backbones.utils.utils import round_channels
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.models.modules.blocks.inverted_residual import InvertedResidualBlock
from src.models.backbones.utils.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN


def _cfg(url='', **kwargs):
    return {
        'url': url, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
    }


default_cfgs = {
    'efficientnet_b0': _cfg(url=''),
    'efficientnet_b1': _cfg(url=''),
    'efficientnet_b2': _cfg(url=''),
    'efficientnet_b3': _cfg(url=''),
    'efficientnet_b4': _cfg(url=''),
    'efficientnet_b5': _cfg(url=''),
    'efficientnet_b6': _cfg(url=''),
    'efficientnet_b7': _cfg(url='')
}

cfg_cls = dict(
    efficientnet_b0=dict(
        WIDTH_COEFFICIENT=1.0,
        DEPTH_COEFFICIENT=1.0,
        DROPOUT_RATE=0.2
    ),
    efficientnet_b1=dict(
        WIDTH_COEFFICIENT=1.0,
        DEPTH_COEFFICIENT=1.1,
        DROPOUT_RATE=0.2
    ),
    efficientnet_b2=dict(
        WIDTH_COEFFICIENT=1.1,
        DEPTH_COEFFICIENT=1.2,
        DROPOUT_RATE=0.3
    ),
    efficientnet_b3=dict(
        WIDTH_COEFFICIENT=1.2,
        DEPTH_COEFFICIENT=1.4,
        DROPOUT_RATE=0.3
    ),
    efficientnet_b4=dict(
        WIDTH_COEFFICIENT=1.4,
        DEPTH_COEFFICIENT=1.8,
        DROPOUT_RATE=0.4
    ),
    efficientnet_b5=dict(
        WIDTH_COEFFICIENT=1.6,
        DEPTH_COEFFICIENT=2.2,
        DROPOUT_RATE=0.4
    ),
    efficientnet_b6=dict(
        WIDTH_COEFFICIENT=1.8,
        DEPTH_COEFFICIENT=2.6,
        DROPOUT_RATE=0.5
    ),
    efficientnet_b7=dict(
        WIDTH_COEFFICIENT=2.0,
        DEPTH_COEFFICIENT=3.1,
        DROPOUT_RATE=0.5
    )
)


class EfficientNet(BaseModel):
    """ (Generic) EfficientNet model."""
    base_model = [
        # expand_ratio, channels, repeats, stride, kernel_size
        [1, 16, 1, 1, 3],
        [6, 24, 2, 2, 3],
        [6, 40, 2, 2, 5],
        [6, 80, 3, 2, 3],
        [6, 112, 3, 1, 5],
        [6, 192, 4, 2, 5],
        [6, 320, 1, 1, 3]
    ]

    def __init__(self, cfg: Dict[str, Any], in_chans: int = 3):
        """Init EfficientNet.

        Args:
            cfg: Model config.
            in_chans: Input channels.
        """
        super().__init__()
        self.in_chans = in_chans
        width_coefficient, depth_coefficient = cfg['WIDTH_COEFFICIENT'], cfg['DEPTH_COEFFICIENT']
        self.last_channels = round_channels(1280, width_coefficient)
        self.blocks = self.__create_blocks(width_coefficient, depth_coefficient)

    def __create_blocks(self, width_coefficient: float, depth_coefficient: float) -> nn.Sequential:
        """Create backbone.

        Args:
            width_coefficient: Channels multiplier.
            depth_coefficient: Layer repeat multiplier.
        """
        in_channels = round_channels(32, width_coefficient, 2)
        blocks = [ConvBnAct(self.in_chans, in_channels, 3, stride=2, padding=1, act_layer=nn.SiLU)]

        for expand_ratio, channels, repeats, stride, kernel_size in self.base_model:
            out_channels = round_channels(channels, width_coefficient, divisor=4)
            layers_repeats = int(math.ceil(repeats * depth_coefficient))

            for layer in range(layers_repeats):
                blocks.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        drop_connect_rate=0.2
                    )
                )
                in_channels = out_channels

        blocks.append(
            ConvBnAct(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        x = self.blocks(x)
        return x

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.last_channels


def create_effnet(variant: str, pretrained: bool = False, **model_kwargs):
    """Create EfficientNet base model.

    Args:
        variant: Backbone type.
        pretrained: If True the pretrained weights will be loaded.
        model_kwargs: Kwargs for model (for example in_chans).
    """
    model = build_model_with_cfg(
        EfficientNet, pretrained, default_cfg=default_cfgs[variant],
        model_cfg=cfg_cls[variant], **model_kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b0(pretrained: bool = False, **kwargs):
    """EfficientNet-B0 """
    model = create_effnet('efficientnet_b0', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b1(pretrained: bool = False, **kwargs):
    """EfficientNet-B1 """
    model = create_effnet('efficientnet_b1', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b2(pretrained: bool = False, **kwargs):
    """EfficientNet-B2 """
    model = create_effnet('efficientnet_b2', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b3(pretrained: bool = False, **kwargs):
    """EfficientNet-B3 """
    model = create_effnet('efficientnet_b3', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b4(pretrained: bool = False, **kwargs):
    """EfficientNet-B4 """
    model = create_effnet('efficientnet_b4', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b5(pretrained: bool = False, **kwargs):
    """EfficientNet-B5 """
    model = create_effnet('efficientnet_b5', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b6(pretrained: bool = False, **kwargs):
    """EfficientNet-B6 """
    model = create_effnet('efficientnet_b6', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def efficientnet_b7(pretrained: bool = False, **kwargs):
    """EfficientNet-B7 """
    model = create_effnet('efficientnet_b7', pretrained=pretrained, **kwargs)
    return model
