"""TorchOK Squeeze-and-Excitation Channel Attention module.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Optional

from torch import nn as nn

from src.models.backbones.utils.utils import make_divisible


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
            reduction_channels = make_divisible(channels * reduction_ratio, divisor, min_channels)
        else:
            reduction_channels = make_divisible(channels // reduction, divisor, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        """Forward method."""
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
