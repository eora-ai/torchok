"""TorchOK Squeeze-and-Excitation Channel Attention module.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Optional, Union

from torch import nn as nn


class SEModule(nn.Module):
    """SE Module as defined in original SE-Nets with a few additions."""

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
        super().__init__()

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

    def __make_divisible(self, value: Union[int, float], divisor: int = 8, min_value: Optional[int] = None):
        """Make divisible function.

        This function rounds the channel number to the nearest value that can be divisible by the divisor.

        Args:
            value: The original channel number.
            divisor: The divisor to fully divide the channel number.
            min_value: The minimum value of the output channel.
        """
        min_value = min_value or divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def forward(self, x):
        """Forward method."""
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
