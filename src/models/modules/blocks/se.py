"""TorchOK Squeeze-and-Excitation Channel Attention module.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Optional

from torch import nn as nn


class SEModule(nn.Module):
    """SE Module as defined in original SE-Nets with a few additions.

    Additions include:
        * min_channels can be specified to keep reduced channel count at a minimum (default: 8)
        * divisor can be specified to keep channels rounded to specified values (default: 1)
        * reduction channels can be specified directly by arg (if reduction_channels is set)
        * reduction channels can be specified by float ratio (if reduction_ratio is set)
    """

    def __init__(self,
                 channels: int,
                 reduction: int = 16,
                 reduction_ratio: Optional[float] = None,
                 reduction_channels: Optional[int] = None,
                 min_channels: int = 8,
                 divisor: int = 1):
        """Init SEModule.

        Args:
            channels: Number input channels.
            reduction: Reduction coefficient.
            reduction_ratio: Reduction ratio.
            reduction_channels: Reduction channels.
            min_channels: Minimum number of reduction channels.
            divisor: `reduction_channels` must be a multiple of `divisor`.
        """
        super(SEModule, self).__init__()

        if reduction_channels is not None:
            reduction_channels = reduction_channels
        elif reduction_ratio is not None:
            reduction_channels = self.__make_divisible(channels * reduction_ratio, divisor, min_channels)
        else:
            reduction_channels = self.__make_divisible(channels // reduction, divisor, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def __make_divisible(self, v, divisor=8, min_value=None):
        min_value = min_value or divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        """Forward method."""
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
