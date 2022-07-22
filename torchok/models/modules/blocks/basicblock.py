from typing import Optional

import torch.nn as nn
from torch import Tensor

from torchok.models.modules.bricks.convbnact import ConvBnAct


class BasicBlock(nn.Module):
    """BasicBlock bulding block for ResNet architecture."""

    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attn_layer: Optional[nn.Module] = None):
        """Init BasicBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride.
            downsample: Downsample module.
            attn_layer: Attention block. SEModule or None.
        """
        super().__init__()
        out_block_channels = out_channels * self.expansion

        self.convbnact1 = ConvBnAct(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.convbnact2 = ConvBnAct(out_channels, out_block_channels, kernel_size=3, padding=1, act_layer=None)
        self.se = attn_layer(out_block_channels) if attn_layer is not None else None
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        identity = x

        out = self.convbnact1(x)
        out = self.convbnact2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
