import torch
import torch.nn as nn

from torchok.constructor import POOLINGS
from torchok.models.poolings.classification.pooling import Pooling


@POOLINGS.register_class
class PoolingLinear(Pooling):
    def __init__(self, in_channels, out_channels, pooling_type: str = 'avg', output_size: int = 1, bias=True):
        super().__init__(in_channels, pooling_type, output_size=output_size)
        self.fc = nn.Linear(self._out_channels, out_channels, bias=bias)
        self._out_channels = out_channels
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.fc(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
