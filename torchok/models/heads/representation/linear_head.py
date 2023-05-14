from typing import Optional

import torch.nn.functional as F
from lightly.models.modules import SwaVProjectionHead as LightlySwaVProjectionHead
from torch import nn, Tensor

from torchok.constructor import HEADS
from torchok.models.base import BaseModel


@HEADS.register_class
class LinearHead(BaseModel):
    """Linear Head"""

    def __init__(self, in_channels, out_channels, drop_rate=0.0, bias=True, normalize=False):
        """Init LinearHead.
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            drop_rate: Drop rate.
            bias: Bias.
            normalize: Normalize.
        """
        super().__init__(in_channels, out_channels)
        self.drop_rate = drop_rate
        self.normalize = normalize
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = self.fc(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x


@HEADS.register_class
class SwaVProjectionHead(LightlySwaVProjectionHead):
    """Projection head used for SwaV.

    [0]: SwAV, 2020, https://arxiv.org/abs/2006.09882
    """

    def __init__(self, in_channels: int = 2048, hidden_channels: int = 2048, out_channels: int = 128):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        super().__init__(input_dim=in_channels, hidden_dim=hidden_channels, output_dim=out_channels)
