import torch.nn as nn
from torch import Tensor

from src.models.modules.blocks.se import SEModule
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.backbones.utils.utils import drop_connect, round_channels


class InvertedResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 expand_ratio: int = None,
                 expand_channels: int = None,
                 act_layer: nn.Module = nn.SiLU,
                 use_se: bool = True,
                 reduction_divisor: int = 2,
                 drop_connect_rate: float = 0.2):
        """Init InvertedResidualBlock.

        Args:
            in_channels: Number of channels.
            out_channels: Number of channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            expand_ratio: Expand ratio.
            act_layer: Activation layer.(default nn.SiLU)
            reduction: Reducton for SEModule.
            drop_connect_rate: Drop connect rate.
        """
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        hidden_dim = round_channels(in_channels, expand_ratio, divisor=2)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        reduction_channels = round_channels(in_channels // 4, divisor=reduction_divisor)
        self.use_expand_block = expand_channels is not None and expand_channels != in_channels or\
                                expand_ratio is not None and expand_ratio != 1

        layers = []

        if self.use_expand_block:
            layers.append(ConvBnAct(in_channels, hidden_dim, kernel_size=1, padding=0, act_layer=act_layer))

        layers.append(
            ConvBnAct(hidden_dim, hidden_dim, kernel_size, padding, stride, groups=hidden_dim, act_layer=act_layer))

        if use_se:
            layers.append(SEModule(hidden_dim, reduction_channels=reduction_channels))

        layers.append(ConvBnAct(hidden_dim, out_channels, 1, 1, 0, bias=False, act_layer=None))

        self.inverted_residual = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + drop_connect(self.inverted_residual(x), self.drop_connect_rate, self.training)
        else:
            return self.inverted_residual(x)
