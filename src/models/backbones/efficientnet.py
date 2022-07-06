"""TorchOK EffcientNet.
Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
import math
from typing import Union, List

import torch.nn as nn
from torch import Tensor

from src.constructor import BACKBONES
from src.models.base_model import BaseModel
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.backbones.utils.utils import round_channels
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.models.modules.blocks.inverted_residual import InvertedResidualBlock

default_cfgs = {
    'efficientnet_b0': dict(url=''),
    'efficientnet_b1': dict(url=''),
    'efficientnet_b2': dict(url=''),
    'efficientnet_b3': dict(url=''),
    'efficientnet_b4': dict(url=''),
    'efficientnet_b5': dict(url=''),
    'efficientnet_b6': dict(url=''),
    'efficientnet_b7': dict(url='')
}

cfg_cls = dict(
    efficientnet_b0=dict(
        width_coefficient=1.0,
        depth_coefficient=1.0
    ),
    efficientnet_b1=dict(
        width_coefficient=1.0,
        depth_coefficient=1.1
    ),
    efficientnet_b2=dict(
        width_coefficient=1.1,
        depth_coefficient=1.2
    ),
    efficientnet_b3=dict(
        width_coefficient=1.2,
        depth_coefficient=1.4
    ),
    efficientnet_b4=dict(
        width_coefficient=1.4,
        depth_coefficient=1.8
    ),
    efficientnet_b5=dict(
        width_coefficient=1.6,
        depth_coefficient=2.2
    ),
    efficientnet_b6=dict(
        width_coefficient=1.8,
        depth_coefficient=2.6
    ),
    efficientnet_b7=dict(
        width_coefficient=2.0,
        depth_coefficient=3.1
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

    def __init__(self, width_coefficient, depth_coefficient, in_chans: int = 3):
        """Init EfficientNet.

        Args:
            width_coefficient: Channels multiplier.
            depth_coefficient: Layer repeat multiplier.
            in_chans: Input channels.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_channels = round_channels(1280, width_coefficient)
        self.blocks = self.__create_blocks(width_coefficient, depth_coefficient)

    def __create_blocks(self, width_coefficient: float, depth_coefficient: float) -> nn.Sequential:
        """Create backbone.

        Args:
            width_coefficient: Channels multiplier.
            depth_coefficient: Layer repeat multiplier.
        """
        in_channels = round_channels(32, width_coefficient, 8)
        blocks = [ConvBnAct(self.in_chans, in_channels, 3, stride=2, padding=1, act_layer=nn.SiLU)]

        for expand_ratio, channels, repeats, stride, kernel_size in self.base_model:
            out_channels = round_channels(channels, width_coefficient, divisor=8)
            layers_repeats = int(math.ceil(repeats * depth_coefficient))

            for layer in range(layers_repeats):
                # generate new reduction_channels every step
                se_kwargs = dict(reduction_channels=round_channels(in_channels//4, divisor=2))
                blocks.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        drop_connect_rate=0.2,
                        se_kwargs = se_kwargs
                    )
                )
                in_channels = out_channels

        blocks.append(
            ConvBnAct(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        x = self.blocks(x)
        return x

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.out_channels


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
