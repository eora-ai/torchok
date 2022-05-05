"""TorchOK Squeeze-and-Excitation Channel Attention module.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Optional

from torch import nn as nn

from src.models.modules.utils.create_act import create_act_layer


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions.

    Additions include:
        * min_channels can be specified to keep reduced channel count at a minimum (default: 8)
        * divisor can be specified to keep channels rounded to specified values (default: 1)
        * reduction channels can be specified directly by arg (if reduction_channels is set)
        * reduction channels can be specified by float ratio (if reduction_ratio is set)
    """

    def __init__(self,
                 channels: list,
                 reduction: int = 16,
                 act_layer: nn.Module = nn.ReLU,
                 gate_layer: str = 'sigmoid',
                 reduction_ratio: Optional[float] = None,
                 reduction_channels: Optional[int] = None,
                 min_channels: int = 8,
                 divisor: int = 1):

        super(SEModule, self).__init__()

        if reduction_channels is not None:
            reduction_channels = reduction_channels
        elif reduction_ratio is not None:
            reduction_channels = self.__make_divisible(channels * reduction_ratio, divisor, min_channels)
        else:
            reduction_channels = self.__make_divisible(channels // reduction, divisor, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def __make_divisible(self, v, divisor=8, min_value=None):
        min_value = min_value or divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation.

    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self,
                 channels: int,
                 gate_layer: str = 'hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer, inplace=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)
