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
                 expand_ratio: int = None,
                 expand_channels: int = None,
                 act_layer: nn.Module = nn.SiLU,
                 use_se: bool = True,
                 se_kwargs: dict = None,
                 drop_connect_rate: float = 0.2):
        """Init InvertedResidualBlock.

        Args:
            in_channels: Number of channels.
            out_channels: Number of channels.
            kernel_size: Kernel size.
            stride: Stride.
            expand_ratio: Expand ratio.
            expand_channels: Expand channels.
            act_layer: Activation layer.(default nn.SiLU)
            use_se: If True will use SEModule.
            se_kwargs: SEModule kwargs.
            drop_connect_rate: Drop connect rate.
        """
        super().__init__()
        se_kwargs = {} if se_kwargs is None else se_kwargs
        self.drop_connect_rate = drop_connect_rate
        expand_channels = round_channels(in_channels, expand_ratio, divisor=2) if expand_channels is None else expand_channels
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.use_expand_block = expand_channels != in_channels

        layers = []

        if self.use_expand_block:
            layers.append(ConvBnAct(in_channels, expand_channels, kernel_size=1, padding=0, act_layer=act_layer))

        layers.append(
            ConvBnAct(expand_channels, expand_channels, kernel_size, kernel_size // 2,
                      stride, groups=expand_channels, act_layer=act_layer))

        if use_se:
            layers.append(SEModule(expand_channels, **se_kwargs))

        layers.append(ConvBnAct(expand_channels, out_channels, 1, 0, 1, bias=False, act_layer=None))

        self.inverted_residual = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + drop_connect(self.inverted_residual(x), self.drop_connect_rate, self.training)
        else:
            return self.inverted_residual(x)
