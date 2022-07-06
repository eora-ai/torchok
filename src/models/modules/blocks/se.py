"""TorchOK Squeeze-and-Excitation Channel Attention module.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Optional

from torch import nn as nn
import torch.nn.functional as F

from src.models.backbones.utils.utils import make_divisible
from src.models.modules.bricks.convbnact import ConvBnAct


class SEModule(nn.Module):
    """SE Module as defined in original SE-Nets with a few additions."""

    def __init__(self,
                 channels: int,
                 reduction: int = 16,
                 reduction_ratio: Optional[float] = None,
                 reduction_channels: Optional[int] = None,
                 min_channels: int = 8,
                 divisor: int = 1,
                 use_pooling: bool = False,
                 use_norm: bool = False,
                 bias: bool = False,
                 gate: nn.Module = nn.Sigmoid):
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
        self.use_pooling = use_pooling
        if reduction_channels is not None:
            reduction_channels = reduction_channels
        elif reduction_ratio is not None:
            reduction_channels = make_divisible(channels * reduction_ratio, divisor, min_channels)
        else:
            reduction_channels = make_divisible(channels // reduction, divisor, min_channels)

        self.convbnact1 = ConvBnAct(channels, reduction_channels, 1, 0, use_norm=use_norm, bias=bias)
        self.convbnact2 = ConvBnAct(reduction_channels, channels, 1, 0, use_norm=use_norm, act_layer=None, bias=bias)
        self.gate = gate()

    def forward(self, x):
        """Forward method."""
        if self.use_pooling:
            x_se = F.adaptive_avg_pool2d(x, 1)
        else:
            x_se = x.mean((2, 3), keepdim=True)
        x_se = self.convbnact1(x_se)
        x_se = self.convbnact2(x_se)
        return x * self.gate(x_se)
