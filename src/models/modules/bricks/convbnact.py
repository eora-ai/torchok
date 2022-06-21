"""TorchOK ConvBnAct module."""
from typing import Optional

import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    """Combination of convolution, batchnorm and activation layers."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 stride: int = 1,
                 bias: bool = False,
                 use_norm: bool = True,
                 groups: int = 1,
                 act_layer: Optional[nn.Module] = nn.ReLU):
        """Init ConvBnAct.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
                default: 1
            bias: Bias.
                default: False
            act_layer: Activation layer.
                default: relu.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if use_norm else None
        self.act = act_layer(inplace=True) if act_layer is not None else None

    def forward(self, x: torch.Tensor):
        """Forward method."""
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x
