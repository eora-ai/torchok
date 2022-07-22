from typing import Optional

import torch.nn as nn
from torch import Tensor

from torchok.models.modules.bricks.convbnact import ConvBnAct


class Bottleneck(nn.Module):
    """Bottleneck building block for ResNet architecture."""

    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attn_layer: Optional[nn.Module] = None):
        """Init Bottleneck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride.
            downsample: Downsample module.
            attn_layer: Attention block.
        """
        super().__init__()
        out_block_channels = out_channels * self.expansion

        self.convbnact1 = ConvBnAct(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.convbnact2 = ConvBnAct(out_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.convbnact3 = ConvBnAct(out_channels, out_block_channels, kernel_size=1, padding=0, act_layer=None)
        self.act = nn.ReLU(inplace=True)
        self.se = attn_layer(out_block_channels) if attn_layer is not None else None
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        identity = x

        out = self.convbnact1(x)
        out = self.convbnact2(out)
        out = self.convbnact3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
